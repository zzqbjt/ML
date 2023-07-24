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

print("Start")
start = time.time() #计时

class_num = 0
class_list = []

class LeavesDataset(Dataset): #数据集类
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

class AlexNet(torch.nn.Module): #神经网络
   def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 96, kernel_size=11, padding=1, stride=4)
        self.conv2=nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3=nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4=nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5=nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bat2d=nn.BatchNorm2d(4)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat=nn.Flatten()
        self.linear1=nn.Linear(5*5*256, 1024)
        self.linear2=nn.Linear(1024, 1024)
        self.linear3=nn.Linear(1024, 176)
        self.dropout=nn.Dropout(p=0.5)
   def forward(self,x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = self.pool(y)
        y = self.flat(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear3(y)
        return y

class ResNet(torch.nn.Module):
    def __init__(self, chanels_num, blocks_num):
        super().__init__()
        self.conv0 = nn.Conv2d(3, chanels_num, kernel_size=7, stride=2, padding=3)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(chanels_num)
        self.BN = blocks_num
        for i in range(blocks_num):
            self.add_module('block'+str(i), nn.Sequential(nn.Conv2d(chanels_num*(2**i), chanels_num*(2**(i+1)), 3, padding=1, stride=2),
                                             nn.BatchNorm2d(chanels_num*(2**(i+1))),
                                             nn.ReLU(),
                                             nn.Conv2d(chanels_num*(2**(i+1)), chanels_num*(2**(i+1)), 3, padding=1, stride=1),
                                             nn.BatchNorm2d(chanels_num*(2**(i+1)))).cuda())
            self.add_module('conv11'+str(i) ,nn.Conv2d(chanels_num*(2**i), chanels_num*(2**(i+1)), 1, stride=2).cuda())
            
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(chanels_num*(2**blocks_num), 176))
        self.relu = nn.ReLU()
    def forward(self, x):
        Y = self.conv0(x)
        Y = self.bn0(Y)
        Y = self.pool0(Y)
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

#1. 导入数据集
path = 'ML\leaves\classify-leaves\\' #读取训练集
datalist = pd.read_csv(path+'train.csv')

L = len(datalist) #划分测试集
l = int(L*0.7)
trainlist = datalist[['image','label']][0:l]
testlist = datalist[['image','label']][l:L]

trans_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.64, 1), ratio=(0.5, 2)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
trans_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

trainIL = []
trainLL = []
for i in range(len(trainlist)):
    img = Image.open(path+trainlist['image'][i])
    img = trans_train(img)
    trainIL.append(img)
    trainLL.append(trainlist['label'][i])

testIL = []
testLL = []
for i in range(l, l+len(testlist)):
    img = Image.open(path+testlist['image'][i])
    img = trans_test(img)
    testIL.append(img)
    testLL.append(testlist['label'][i])

#bagging算法，随机采样
trainIL2 = []
trainLL2 = []
for i in range(int(0.5*len(trainIL))):
    x = random.randint(0, len(trainIL)-1)
    trainIL2.append(trainIL[x])
    trainLL2.append(trainLL[x])
trainIL = trainIL2
trainLL = trainLL2

class_list = list(np.unique(trainLL))
class_num = len(np.unique(trainLL))
for i in range(len(trainLL)):
    trainLL[i] = class_list.index(trainLL[i])
for i in range(len(testLL)):
    testLL[i] = class_list.index(testLL[i])


#2. 模型设置
learning_rate_max = 1e-2
learning_rate_min = 1e-3
train_batch_size = 64  #指定DataLoader在训练集中每批加载的样本数量
test_batch_size = 64  #指定DataLoader在测试集中每批加载的样本数量
num_epoches = 50 #迭代次数
mom = 0.9 #设置SGD中的冲量

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
dataset = LeavesDataset(imglist = trainIL,
                          labellist = trainLL)                  
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

dataset = LeavesDataset(imglist = testIL,
                          labellist = testLL)                     
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=True)
stage1 = time.time()
print("Input:", stage1-start, 's')

model = ResNet(64, 3)

# state_dict = torch.load(path+'leaves_net_params.pth') 
# model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate_max, momentum=mom) #随机梯度下降

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
    for sample in train_dataloader:
        img = sample['image']
        label = sample['label']
        
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

        for sample in test_dataloader:
            img = sample['image']
            label = sample['label']
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = criterion(out, label) #记录误差
            eval_loss += loss.item() #记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
        
        eval_losses.append(eval_loss / len(test_dataloader))
        eval_acces.append(eval_acc / len(test_dataloader))
        curr = time.time()
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'
                .format(epoch, train_loss / len(train_dataloader), train_acc / len(train_dataloader), 
                eval_loss / len(test_dataloader), eval_acc / len(test_dataloader), curr-prev))
        if eval_acc / len(test_dataloader) > best_acc:
            best_acc = eval_acc / len(test_dataloader)
            torch.save(model.state_dict(), path+'leaves_net_params1.pth')

#5. 误差、正确率随时间变化图像
# x = range(0, num_epoches)
# x = np.array(x)
# losses = np.array(losses)
# acces = np.array(acces)
# eval_losses = np.array(eval_losses)
# eval_acces = np.array(eval_acces)
# plt.figure(figsize=(20, 20), dpi=100)
# l1, = plt.plot(x, losses, c='red')
# l2, = plt.plot(x, acces, c='blue')
# l3, = plt.plot(x, eval_losses, c='green')
# l4, = plt.plot(x, eval_acces, c='orange')
# plt.legend(handles=[l1,l3],labels=['Training Losses','Test Losses'], prop = {'size': 20}, loc='best')
# plt.xlabel("Training Time", fontdict={'size': 20})
# plt.ylabel("Value", fontdict={'size': 16})
# plt.title("Training Process", fontdict={'size': 30})
# plt.savefig(path+'训练过程3-loss.png',dpi=100)
# plt.legend(handles=[l2,l4],labels=['Training Acces','Test Acces'], prop = {'size': 20}, loc='best')
# plt.xlabel("Training Time", fontdict={'size': 20})
# plt.ylabel("Value", fontdict={'size': 16})
# plt.title("Training Process", fontdict={'size': 30})
#plt.savefig(path+'训练过程3-acc.png',dpi=100)

end = time.time()
print("End:", end-start, 's')
