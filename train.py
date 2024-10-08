import torch
import torch.nn as nn
import torch.optim as optim
from mobilevit import MobileViT
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def mobilevit_pedalkeeper():
    print('[mobilevit_pedalkeeper] init')
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1)

def Train(model, video_scene, pedal_scene, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[Train] Using device:', device)
    
    criterion = nn.MSELoss()  # 예시: 분류 문제
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    video_tensor = torch.Tensor([video_scene]).to(device)
    pedal_tensor = torch.Tensor([pedal_scene]).to(device)

    model.to(device)

    for epoch in range(num_epochs):
        for i in range(len(video_scene)):
            # 데이터 전처리 (필요한 경우)

            # 순전파
            # output = model(torch.Tensor([[video_scene[i][0:480, 80:560]]]))
            # 256 256
            optimizer.zero_grad()
            output = model(torch.Tensor([[video_scene[i][112:368, 192:448]]]).to(device))
            loss = criterion(output, torch.Tensor([[pedal_scene[i]]]).to(device))

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            # 학습 과정 출력 (선택 사항)
            if i % 100 == 0:
                print(f"[Train] Epoch [{epoch}/{num_epochs}], Scene [{i}/{len(video_scene)}], Loss: {loss.item():.4f}")
                
    return model

def Verify(model, video_scene, pedal_scene):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0

    with torch.no_grad():
        for i in range(len(video_scene)):
            # 데이터 전처리 (필요한 경우)

            # 순전파
            # output = model(torch.Tensor([[video_scene[i][0:480, 80:560]]]))
            # 256 256
            output = model(torch.Tensor([[video_scene[i][112:368, 192:448]]]))
            if output > 0.5 and pedal_scene[i] > 0.5:
                correct += 1
            elif output < 0.5 and pedal_scene[i] < 0.5:
                correct += 1

    accuracy = 100 * correct / len(video_scene)
    print(f'[Verify] Accuracy on the test set: {accuracy:.2f}%')