import torch
import torchvision
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
import time
import matplotlib.pyplot as plt

print("Start")
start = time.time()
torch.set_default_dtype(torch.float64)
class MLP(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        self.layers_num = len(hidden) + 1 
        hidden.append(output)
        hidden.insert(0, input)
        for i in range(len(hidden)-1):
            if i != len(hidden)-2:
                self.add_module('linear'+str(i), nn.Sequential(
                    nn.Linear(hidden[i], hidden[i+1]),
                    nn.ReLU(),
                    nn.Dropout(p = 0.5)
                ))
            else:
                self.add_module('linear'+str(i), nn.Sequential(
                    nn.Linear(hidden[i], hidden[i+1]),
                ))
    def forward(self, x):
        y = x
        for i in range(self.layers_num):
            func = getattr(self, 'linear'+str(i))
            y = func(y)
        return y

class ResNet(torch.nn.Module):
    def __init__(self, chanels_num, layers_num, blocks_num, num_classes):
        super().__init__()
        self.conv0 = nn.Conv1d(1, chanels_num, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm1d(chanels_num)
        self.BN = blocks_num
        self.LN = layers_num
        for i in range(layers_num):
            self.add_module('layer'+str(i), nn.Sequential(nn.Conv1d(chanels_num, chanels_num, 3, padding=1, stride=1),
                                             nn.BatchNorm1d(chanels_num),
                                             nn.ReLU(),
                                             nn.Conv1d(chanels_num, chanels_num, 3, padding=1, stride=1),
                                             nn.BatchNorm1d(chanels_num)).cuda())
        for i in range(blocks_num):
            self.add_module('block'+str(i), nn.Sequential(nn.Conv1d(chanels_num*(2**i), chanels_num*(2**(i+1)), 3, padding=1, stride=2),
                                             nn.BatchNorm1d(chanels_num*(2**(i+1))),
                                             nn.ReLU(),
                                             nn.Conv1d(chanels_num*(2**(i+1)), chanels_num*(2**(i+1)), 3, padding=1, stride=1),
                                             nn.BatchNorm1d(chanels_num*(2**(i+1)))).cuda())
            self.add_module('conv11'+str(i), nn.Sequential(
                                            nn.Conv1d(chanels_num*(2**i), chanels_num*(2**(i+1)), 1, stride=2),
                                            nn.BatchNorm1d(chanels_num*(2**(i+1)))))
      
        self.pool1 = nn.AdaptiveAvgPool1d(1)
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

class MyDataset(Dataset): #数据集类
    def __init__(self, datalist, labellist, transform=None):
        self.transform = transform
        self.datalist = datalist
        self.labellist = labellist
        self.size = len(datalist)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.datalist[idx]
        label = self.labellist[idx]
        if self.transform:
           data = self.transform(data)
        sample = {'data': data, 'label': label}
        return sample


path = './ML/kaggle/Spaceship Titanic/'
data = pd.read_csv(path + 'train.csv')

Fdata = []
Flabel = []
null = -1
L = len(data)
l = int(0.9*L)
for i in range(L):
    group = int(data['PassengerId'][i][0:4])

    HomePlanet = data['HomePlanet'][i]
    if HomePlanet == 'Earth':
        HomePlanet = 0  
    elif HomePlanet == 'Europa':
        HomePlanet = 1
    elif HomePlanet == 'Mars':
        HomePlanet = 2
    else:
        HomePlanet = null

    try:
        CryoSleep = int(data['CryoSleep'][i])
    except:
        CryoSleep = null

    try:
        CabinDeck, CabinNum, CabinSide = data['Cabin'][i].split('/')
        CabinNum = int(CabinNum)
        CabinDeck = ord(CabinDeck) - ord('A') + 1
        CabinSide = (0 if CabinSide == 'P' else 1)
    except:
        CabinDeck, CabinNum, CabinSide = null, null, null
        
    Destination = data['Destination'][i]
    if Destination == 'TRAPPIST-1e':
        Destination = 0  
    elif Destination == 'PSO J318.5-22':
        Destination = 1
    elif Destination == '55 Cancri e':
        Destination = 2
    else:
        Destination = null

    try:
        Age = int(data['Age'][i])
    except:
        Age = null

    try:
        VIP = int(data['VIP'][i])
    except:
        VIP = null

    try:
        RoomService	= int(data['RoomService'][i])
    except:
        RoomService = null

    try:
        FoodCourt = int(data['FoodCourt'][i])
    except:
        FoodCourt = null

    try:
        ShoppingMall = int(data['ShoppingMall'][i])
    except:
        ShoppingMall = null	

    try:
        Spa = int(data['Spa'][i])
    except:
        Spa = null	

    try:
        VRDeck = int(data['VRDeck'][i])	
    except:
        VRDeck = null	

    Transported = int(data['Transported'][i])	

    Fdata.append([group, HomePlanet, CryoSleep, CabinDeck, CabinNum, CabinSide, Destination, Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck])
    Flabel.append(Transported)

Fdata = np.array(Fdata)
Fdata = preprocessing.scale(Fdata)
Flabel = torch.tensor(Flabel)
Flabel = Flabel.long()

# model = MLP(13, 2, [100])
model = ResNet(16, 1, 1, 2)
if torch.cuda.is_available():
    model = model.cuda()
# state_dict = torch.load(path+'params.pth') 
# model.load_state_dict(state_dict)

learning_rate_max = 0.1
learning_rate_min = 0.01
num_epoches = 50
mom = 0.9
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=learning_rate_max, momentum=mom)

stage1 = time.time()
print("Input:", stage1-start, 's')

train_data = Fdata[0:l, :]
train_label = Flabel[0:l]  

valid_data = Fdata[l:L, :]
valid_label = Flabel[l:L]

train_data = torch.tensor(train_data)
valid_data = torch.tensor(valid_data)

train_dataset = MyDataset(datalist = train_data,
                        labellist = train_label)                  
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = MyDataset(datalist = valid_data,
                        labellist = valid_label)                  
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

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
        Data = sample['data']
        label = sample['label']
        Data = Data.view(Data.shape[0], 1, -1)

        Data = Data.to(device)
        label = label.to(device)
        Data = Data.view(Data.shape[0], 1, -1)

        
        out = model(Data)  #前向传播
        print(Data.requires_grad, out.requires_grad, label.requires_grad)
        loss = criterion(out, label)  

        optimizer.zero_grad() #反向传播
        loss.backward()  
        optimizer.step() 
        
        train_loss += loss.item()  
        _, pred = out.max(1)

        num_correct = (pred == label).sum().item()
        acc = num_correct / Data.shape[0]  
        train_acc += acc
        
    losses.append(train_loss / len(train_dataloader)) #所有样本平均损失
    acces.append(train_acc / len(train_dataloader)) #所有样本的准确率
    with torch.no_grad():
#4. 测试
        eval_loss = 0
        eval_acc = 0
        model.eval()

        for sample in valid_dataloader:
            Data = sample['data']
            label = sample['label']
            Data = Data.view(Data.shape[0], 1, -1)

            Data = Data.to(device)
            label = label.to(device)

            out = model(Data)
            loss = criterion(out, label) #记录误差

            eval_loss += loss.item() #记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / Data.shape[0]
            eval_acc += acc
        
        eval_losses.append(eval_loss / len(valid_dataloader))
        eval_acces.append(eval_acc / len(valid_dataloader))
        curr = time.time()
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'
                .format(epoch, train_loss / len(train_dataloader), train_acc / len(train_dataloader), 
                eval_loss / len(valid_dataloader), eval_acc / len(valid_dataloader), curr-prev))   

x = range(0, num_epoches)
x = np.array(x)
losses = np.array(losses)
acces = np.array(acces)
eval_losses = np.array(eval_losses)
eval_acces = np.array(eval_acces)
plt.figure(figsize=(20, 20), dpi=100)
l1, = plt.plot(x, losses, c='red')
l3, = plt.plot(x, eval_losses, c='green')
plt.legend(handles=[l1,l3],labels=['Training Losses','valid Losses'], prop = {'size': 20}, loc='best')
plt.xlabel("Training Time", fontdict={'size': 20})
plt.ylabel("Value", fontdict={'size': 16})
plt.title("Training Process", fontdict={'size': 30})
plt.savefig(path+'mlp训练过程-loss.png',dpi=100)
plt.clf()
l2, = plt.plot(x, acces, c='blue')
l4, = plt.plot(x, eval_acces, c='orange')
plt.legend(handles=[l2,l4],labels=['Training Acces','valid Acces'], prop = {'size': 20}, loc='best')
plt.xlabel("Training Time", fontdict={'size': 20})
plt.ylabel("Value", fontdict={'size': 16})
plt.title("Training Process", fontdict={'size': 30})
plt.savefig(path+'mlp训练过程-acc.png',dpi=100)

torch.save(model.state_dict(), path+'params_conv.pth')
end = time.time()
print("End:", end-start, 's')

    