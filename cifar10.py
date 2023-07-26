import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import os
import d2l
from d2l import torch as d2l

print("Start")
start = time.time() #计时

path = './ML/CIFAR10/'

class ResNet(torch.nn.Module):
    def __init__(self, chanels_num, layers_num, blocks_num, num_classes):
        super().__init__()
        self.conv0 = nn.Conv2d(3, chanels_num, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(chanels_num)
        self.BN = blocks_num
        self.LN = layers_num
        for i in range(layers_num):
            self.add_module('layer'+str(i), nn.Sequential(nn.Conv2d(chanels_num, chanels_num, 3, padding=1, stride=1),
                                             nn.BatchNorm2d(chanels_num),
                                             nn.ReLU(),
                                             nn.Conv2d(chanels_num, chanels_num, 3, padding=1, stride=1),
                                             nn.BatchNorm2d(chanels_num)).cuda())
        for i in range(blocks_num):
            self.add_module('block'+str(i), nn.Sequential(nn.Conv2d(chanels_num*(2**i), chanels_num*(2**(i+1)), 3, padding=1, stride=2),
                                             nn.BatchNorm2d(chanels_num*(2**(i+1))),
                                             nn.ReLU(),
                                             nn.Conv2d(chanels_num*(2**(i+1)), chanels_num*(2**(i+1)), 3, padding=1, stride=1),
                                             nn.BatchNorm2d(chanels_num*(2**(i+1)))).cuda())
            self.add_module('conv11'+str(i), nn.Sequential(
                                            nn.Conv2d(chanels_num*(2**i), chanels_num*(2**(i+1)), 1, stride=2),
                                            nn.BatchNorm2d(chanels_num*(2**(i+1)))))
      
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(chanels_num*(2**blocks_num), num_classes))
        self.relu = nn.ReLU()
    def forward(self, x):
        Y = self.conv0(x)
        Y = self.bn0(Y)
        Y = self.relu(Y)
        for i in range(self.LN):
            X = Y
            func = getattr(self, 'layer'+str(i))
            Y = func(Y)
            Y = Y+X
            Y = self.relu(Y)

        for i in range(self.BN):
            X = Y
            func1 = getattr(self, 'block'+str(i))
            func2 = getattr(self, 'conv11'+str(i))
            Y = func1(Y)
            X = func2(X)
            Y = Y+X
            Y = self.relu(Y)
        Y = self.pool1(Y)
        Y = self.linear(Y)
        return Y

class CIFARDataset(Dataset): #数据集类
    def __init__(self, imglist, labellist, transform=None):
        self.transform = transform
        self.imglist = imglist
        self.labellist = labellist
        self.size = len(imglist)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.imglist[idx]
        label = self.labellist[idx]
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': label}
        return sample

trans_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

trans_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])


learning_rate_max = 0
learning_rate_min = 0
batch_size = 128  
num_epoches = 1
mom = 0.9
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True, 
                                             transform=trans_train,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                            train=False, 
                                            transform=trans_valid)

# 数据载入
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model.conv1 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1)
model.maxpool = nn.Sequential()
nn.init.xavier_uniform_(model.fc.weight)
# model = ResNet(16, 2, 3, 10)
state_dict = torch.load(path+'params\'.pth') 
model.load_state_dict(state_dict)


if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate_max, momentum=0.9) #随机梯度下降

stage1 = time.time()
print("Input:", stage1-start, 's')
#4. 训练
losses = []       #记录训练集损失
acces = []        #记录训练集准确率
eval_losses = []  #记录测试集损失
eval_acces = []  #记录测试集准确率
best_acc = 0
for epoch in range(num_epoches):
    prev = time.time()
    train_loss = 0
    train_acc = 0
    model.train()   
    optimizer.param_groups[0]['lr'] = learning_rate_min+(learning_rate_max-learning_rate_min)/2*(np.cos(epoch/num_epoches*np.pi)+1)
    for img, label in train_dataloader:        
        img = img.to(device)
        label = label.to(device)
        
        out = model(img)  #前向传播
        loss = criterion(out, label)

        optimizer.zero_grad() #反向传播
        loss.backward()  
        optimizer.step() 
        
        train_loss += loss.item()  
        _, pred = out.max(1) 
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]  
        train_acc += acc
        
    losses.append(train_loss / len(train_dataloader)) #所有样本平均损失
    acces.append(train_acc / len(train_dataloader)) #所有样本的准确率
    with torch.no_grad():
#4. 测试
        eval_loss = 0
        eval_acc = 0
        model.eval()

        for img, label in valid_dataloader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = criterion(out, label) #记录误差
            eval_loss += loss.item() #记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
        
        eval_losses.append(eval_loss / len(valid_dataloader))
        eval_acces.append(eval_acc / len(valid_dataloader))
        curr = time.time()
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'
                .format(epoch, train_loss / len(train_dataloader), train_acc / len(train_dataloader), 
                eval_loss / len(valid_dataloader), eval_acc / len(valid_dataloader), curr-prev))
        if eval_acc / len(valid_dataloader) > best_acc:
            best_acc = eval_acc / len(valid_dataloader)
            # torch.save(model.state_dict(), path+'params\'.pth')
# 5. 误差、正确率随时间变化图像
# x = range(0, num_epoches)
# x = np.array(x)
# losses = np.array(losses)
# acces = np.array(acces)
# eval_losses = np.array(eval_losses)
# eval_acces = np.array(eval_acces)
# plt.figure(figsize=(20, 20), dpi=100)
# l1, = plt.plot(x, losses, c='red')
# l3, = plt.plot(x, eval_losses, c='green')
# plt.legend(handles=[l1,l3],labels=['Training Losses','valid Losses'], prop = {'size': 20}, loc='best')
# plt.xlabel("Training Time", fontdict={'size': 20})
# plt.ylabel("Value", fontdict={'size': 16})
# plt.title("Training Process", fontdict={'size': 30})
# plt.savefig(path+'训练过程-loss\'.png',dpi=100)
# plt.clf()
# l2, = plt.plot(x, acces, c='blue')
# l4, = plt.plot(x, eval_acces, c='orange')
# plt.legend(handles=[l2,l4],labels=['Training Acces','valid Acces'], prop = {'size': 20}, loc='best')
# plt.xlabel("Training Time", fontdict={'size': 20})
# plt.ylabel("Value", fontdict={'size': 16})
# plt.title("Training Process", fontdict={'size': 30})
# plt.savefig(path+'训练过程-acc\'.png',dpi=100)

end = time.time()
print("End:", end-start, 's')
