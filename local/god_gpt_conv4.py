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
        return torch.FloatTensor(data).cuda(), label


# 모델 정의
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=129, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2580, 64)
        self.fc2 = nn.Linear(64, num_classes)
        # self.dropout = nn.Dropout(p=0.5)

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
        x = self.conv4(x)
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


# 데이터 로드 및 전처리
input_size = 357
num_classes = 100

# train, test 폴더 경로
train_path = "./train"
test_path = "./val"

# 클래스 정보 추출
train_file_paths = []
train_labels = []
test_file_paths = []
test_labels = []

# train 폴더 내의 파일 경로 및 레이블 정보 추출
subfolders = [f.path for f in os.scandir(train_path) if f.is_dir()]
for subfolder in subfolders:
    class_label = os.path.basename(subfolder)
    csv_files = glob.glob(os.path.join(subfolder, "*.csv"))
    train_file_paths.extend(csv_files)
    train_labels.extend([class_label] * len(csv_files))

# test 폴더 내의 파일 경로 및 레이블 정보 추출
subfolders = [f.path for f in os.scandir(test_path) if f.is_dir()]
for subfolder in subfolders:
    class_label = os.path.basename(subfolder)
    csv_files = glob.glob(os.path.join(subfolder, "*.csv"))
    test_file_paths.extend(csv_files)
    test_labels.extend([class_label] * len(csv_files))

# 클래스를 숫자로 변환
label_map = {label: i for i, label in enumerate(set(train_labels))}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

# 데이터셋 생성
train_dataset = CustomDataset(train_file_paths, train_labels)
test_dataset = CustomDataset(test_file_paths, test_labels)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 인스턴스 생성
model = CNN1D(input_size, num_classes)

# 모델 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1).to(device)  # 차원 추가하여 1D 시계열 데이터로 변환
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

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

    # 가장 높은 정확도를 가진 모델의 가중치 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_acc_weights = model.state_dict()

    # 가장 높은 f1 score를 가진 모델의 가중치 저장
    if f1 > best_f1_score:
        best_f1_score = f1
        best_f1_weights = model.state_dict()

    accuracy_list.append(accuracy)
    f1_score_list.append(f1)

# 가장 높은 정확도와 f1 score를 가진 모델의 가중치 저장
torch.save(best_acc_weights, "best_model_acc_weights.pth")
torch.save(best_f1_weights, "best_model_f1_weights.pth")

# 에폭 종료 후 정확도와 F1 스코어의 평균값 계산
avg_accuracy = total_accuracy / num_epochs
avg_f1_score = total_f1_score / num_epochs

# 에폭 종료 후 정확도와 F1 스코어의 표준편차 계산
accuracy_std = np.std([accuracy for accuracy in accuracy_list])
f1_score_std = np.std([f1 for f1 in f1_score_list])

print("  ")
print(f"Highest Test Accuracy: {best_accuracy:.2f}%")
print(f"Average Test Accuracy: {avg_accuracy:.2f}%")
print(f"Accuracy Standard Deviation: {accuracy_std:.2f}")
print("  ")
print(f"Highest Test F1 Score: {best_f1_score:.2f}%")
print(f"Average Test F1 Score: {avg_f1_score:.2f}%")
print(f"F1 Score Standard Deviation: {f1_score_std:.2f}")
