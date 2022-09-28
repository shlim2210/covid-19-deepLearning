import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import copy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from ipywidgets import interact
from torchsummary import summary

train_data_dir = './Covid19-dataset/train/'
val_data_dir = './Covid19-dataset/test/'
class_list = ['Normal', 'Covid', 'Viral Pneumonia']

def list_image_file(data_dir, sub_dir):
    image_format = ['jpeg', 'jpg', 'png']
    image_files = []
    
    # os.path.join(data_dir, sub_dir) : ./Covid19-dataset/train/Normal, 이미지가 들어있는 폴더의 경로
    images_dir = os.path.join(data_dir, sub_dir)
    # os.listdir(images_dir) : 해당 경로에 들어있는 파일 이름을 리스트로 반환
    for file_path in os.listdir(images_dir):
        # image_format에 해당하는 확장명을 가진 파일이라면
        if file_path.split(".")[-1] in image_format:
            image_files.append(os.path.join(sub_dir, file_path))
    # image_files : 이미지 파일의 경로 리스트 ex) 'Normal\\01.jpeg'
    return image_files

# transform에 여러 단계가 있는 경우, Compose를 통해 여러 단계를 하나로 묶을 수 있습니다. transforms에 속한 함수들을 Compose를 통해 묶어서 한번에 처리할 수 있습니다.
transformer = transforms.Compose([
    # ToTensor : 입력 데이터가 NumPy 배열 또는 PIL 이미지 형식인 경우 ToTensor를 사용하여 텐서 형식으로 변환할 수 있습니다.
    transforms.ToTensor(),
    # transforms.Resize : 이미지의 사이즈 조정
    transforms.Resize((224, 224)),
    # Normalize : Normalize 작업은 텐서를 가져와 평균 및 표준 편차로 정규화합니다. - mean : (sequence)형식으로 평균을 입력하며, 괄호 안에 들어가 있는 수의 개수가 채널의 수. - std : (sequence)형식으로 표준을 입력하며, 괄호 안에 들어가 있는 수의 개수가 채널의 수.
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

# 데이터셋 생성을 위한 클래스
class Covid_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        normals = list_image_file(data_dir, 'Normal')
        covids = list_image_file(data_dir, 'Covid')
        pneumonias = list_image_file(data_dir, 'Viral Pneumonia')
        self.files_path = normals + covids + pneumonias
        self.transform = transform
        
    def __len__(self):
        return len(self.files_path)
        
    def __getitem__(self, index):
        image_file = os.path.join(self.data_dir, self.files_path[index])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = class_list.index(self.files_path[index].split(os.sep)[-2])
        target = class_list.index(self.files_path[index].split(os.sep)[0])
        
        if self.transform:
            image = self.transform(image)
            target = torch.Tensor([target]).long()
        
        return {'image':image, 'target':target}

def covid_dataloader(train_data_dir, val_data_dir):
    dataloaders = {}
    train_dset = Covid_dataset(train_data_dir, transformer)
    # 학습용 데이터로더에서 batch_size를 4로, 학습데이터를 shuffle하고, 마지막 batch의 데이터 수는 다를 수 있기 때문에 drop
    dataloaders['train'] = DataLoader(train_dset, batch_size=4, shuffle=True, drop_last=True)
    
    val_dset = Covid_dataset(val_data_dir, transformer)
    # 검증용 데이터로더는 하나의 batch로 데이터셋 전체를 사용
    dataloaders['val'] = DataLoader(val_dset, batch_size=1, shuffle=False, drop_last=False)
    return dataloaders

# vgg19 딥러닝 모델을 기반으로 함수 생성
# batch size / channel / width / height로 구성된 텐서
def build_covid_model(device_name='cpu'):
    device = torch.device(device_name)
    # vgg19 모델을 불러옴
    model = models.vgg19(pretrained=True)
    # height X width 크기의 텐서를 1x1로 축소, 영역의 평균값을 반환
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Sequential(
        # 다 차원의 텐서를 평탄화시킴
        nn.Flatten(),
        # weights와 bias에 따라 입력값을 선형으로 반환
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(class_list)),
        # 다중 클래스를 분류 할 때 주로 사용
        nn.Softmax(dim=1)
    )
    return model.to(device)


@torch.no_grad()
def get_accuracy(image, target, model):
    # image에서 batch_size 추출
    batch_size = image.shape[0]
    # 검증 데이터로 예측
    prediction = model(image)
    _, pred_label = torch.max(prediction, dim=1)
    # print(pred_label)
    is_correct = (pred_label == target)
    return is_correct.cpu().numpy().sum() / batch_size

# 데이터셋을 train, val에 따라 구분하여 학습을 진행하거나 검증 로직 진행
def train_covid(dataloaders, model, optimizer, loss_func, device):
    losses = {}
    accuracies = {}
    
    for tv in ['train', 'val']:
        running_loss = 0.0
        running_correct = 0
        
        # model.train() : Dropout layer, BatchNorm layer를 사용하는 함수. (train에서 사용). 과적합을 막기 위해 학습에서는 dropout을 사용
        # model.eval() : Dropout layer, BatchNorm layer를 사용하지 않는 함수. (val에서 사용) . 검증할때는 정확한 검증을 위해 dropout을 사용하지 않음
        if tv == 'train':
            model.train()
        else:
            model.eval()
        
        
        # print("dataloaders[tv] : ", len(dataloaders[tv])) 
        # 62 (251:전체 imgs 수 // 4:batch size)
        for index, batch in enumerate(dataloaders[tv]):
            image = batch['image'].to(device) #torch.Size([4, 3, 224, 224])
            #  squeeze(dim=-1) : target이 ([4(batch size)]) 형태이어야 하는데, [Batch size, 1]로 2차원 텐서로 되어 있음. 이걸 1차원 텐서로 변경시켜줘야 함
            target = batch['target'].squeeze(dim=1).to(device)
            
            ### 순전파(Forward Propagation)
            # set_grad_enabled : True, False에 따라 
            with torch.set_grad_enabled(tv == 'train'):
                # 이미지로 예측하여 실제 결과와 비교, 손실함수로 차이를 계산
                prediction = model(image)
                loss = loss_func(prediction, target)
                
                ### 역전파(Back Propagation) : 학습한 정보를 피드백하여 가중치 정보를 변경
                # train dataloader일 경우에만 역전파 진행
                if tv == 'train':
                    # gradient를 0으로 초기화
                    optimizer.zero_grad()
                    # 비용함수를 미분하여 gradient 계산
                    loss.backward()
                    # W와 b를 업데이트
                    optimizer.step()
            
            running_loss += loss.item()
            running_correct += get_accuracy(image, target, model)
            # running_loss : 각 batch(1/62)의 loss 합계
            # running_correct : 각 batch(1/62)의 정확도 합계(0, 0.25, 0.5, 0.75, 1)
            if tv == 'train':
                if index % 10 == 0:
                    print(f"{index}/{len(dataloaders['train'])} - Running loss: {loss.item()}") # 10번마다 해당 batch의 loss값 
        
        # losses[tv] : 학습한 batch들의 평균 loss, 학습한 accuracies[tv]의 평균 accuracy
        losses[tv] = running_loss / len(dataloaders[tv])
        accuracies[tv] = running_correct / len(dataloaders[tv])
    return losses, accuracies

# './trained_model' 경로에 최적화된 학습 모델 저장
def save_best_model(model_state, model_name, save_dir='./trained_model'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_state, os.path.join(save_dir, model_name))


