import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 3번째 컬럼 추출하여 1차원 시계열 데이터로 변환
        data = df.iloc[:, 2].values.astype(float)

        # PyTorch의 FloatTensor로 변환하여 반환
        return torch.FloatTensor(data), label


# 모델 정의
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2688, 64)
        #self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.shape)
        return x

# 데이터 로드 및 전처리
input_size = 357
num_classes = 100

# train, test 폴더 경로
test_path = "./test"

# 클래스 정보 추출
test_file_paths = []
test_labels = []

# test 폴더 내의 파일 경로 및 레이블 정보 추출
subfolders = [f.path for f in os.scandir(test_path) if f.is_dir()]
for subfolder in subfolders:
    class_label = os.path.basename(subfolder)
    csv_files = glob.glob(os.path.join(subfolder, "*.csv"))
    test_file_paths.extend(csv_files)
    test_labels.extend([class_label] * len(csv_files))

# 클래스를 숫자로 변환

# 데이터셋 생성
test_dataset = CustomDataset(test_file_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# 모델 인스턴스 생성
model = CNN1D(input_size, num_classes)

# 모델 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load("./best_acc_weights/best_model_acc_weights_conv3.pth", map_location="cuda:0"))

model.eval()
true_labels = []
pred_labels = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)  # 차원 추가하여 1D 시계열 데이터로 변환
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.tolist())
        pred_labels.extend(predicted.tolist())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
f1 = f1_score(true_labels, pred_labels, average="weighted")

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test F1 Score: {f1:.2f}%")