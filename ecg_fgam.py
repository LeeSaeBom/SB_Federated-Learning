import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import numpy as np
import torch.nn.functional as F


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
        return torch.FloatTensor(data).cuda(), label


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
        # self.dropout = nn.Dropout(p=0.5)
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
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.shape)
        return x


# 가중치 읽어오기
model = CNN1D()
model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

# 데이터 로드 및 전처리
input_size = 357
num_classes = 100

# test 폴더 경로
test_path = "/home/iiplab/Desktop/SB/ECG/val"

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
label_map = {label: i for i, label in enumerate(set(test_labels))}
test_labels = [label_map[label] for label in test_labels]

# 데이터셋 생성
test_dataset = CustomDataset(test_file_paths, test_labels)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_f1_score = 0.0  # 가장 높은 F1 스코어 저장을 위한 변수 초기화
best_accuracy = 0.0  # 가장 높은 정확도 저장을 위한 변수 초기화

num_epochs = 10
total_accuracy = 0.0
total_f1_score = 0.0

# 평가 결과 저장을 위한 리스트
accuracy_list = []
f1_score_list = []

# 모델 평가
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

total_accuracy += accuracy
total_f1_score += f1

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test F1 Score: {f1:.2f}%")


# FGSM 공격
def fgsm_attack(model, data, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output, target)  # 예시로 Negative Log Likelihood (NLL) loss 사용

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()

    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # 입력 데이터의 값을 0~1 범위로 유지

    return perturbed_data


# FGSM 공격 수행
epsilon = 0.03  # 공격 강도를 조절합니다.
perturbed_data = fgsm_attack(model, test_data, epsilon)

# 결과 확인
output = model(perturbed_data.unsqueeze(0))
pred = output.argmax(dim=1)
print("예측 결과:", pred)
