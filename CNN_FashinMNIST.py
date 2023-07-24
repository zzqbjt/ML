import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import time
import cv2
import matplotlib.pyplot as plt
from skimage import io
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
class HDR(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2=nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bat2d=nn.BatchNorm2d(32)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2)
        self.linear=nn.Linear(14 * 14 * 32, 100)
        self.tanh=nn.Tanh()
        self.linear1=nn.Linear(100, 50)
        self.linear2=nn.Linear(50, 10)
   def forward(self,x):
        y=self.conv1(x)
        y=self.bat2d(y)
        y=self.relu(y)
        y=self.pool(y)
        y=y.view(y.size()[0],-1)
        y=self.linear(y)
        y=self.relu(y)
        y=self.linear1(y)
        y=self.relu(y)
        y=self.linear2(y)
        return y

print("Start")
start = time.time() #计时

#1. 超参数
learning_rate = 1e-2
train_batch_size = 64  #指定DataLoader在训练集中每批加载的样本数量
test_batch_size = 128  #指定DataLoader在测试集中每批加载的样本数量
num_epoches = 10 #迭代次数
mom = 0.5 #设置SGD中的冲量
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #GPU加速
#device=torch.device('cpu') 

#2. 读入MNIST
dataset = torchvision.datasets.FashionMNIST('/data', train=True, download=True, #训练集
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),                              
                             ]))                         
train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=train_batch_size,shuffle=True)

dataset = torchvision.datasets.FashionMNIST('/data', train=False, download=True, #测试集
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),                    
                             ]))                       
test_dataloader = torch.utils.data.DataLoader(dataset,batch_size=test_batch_size,shuffle=True)
stage1 = time.time()
print("Input:", stage1-start, 's')

#3. 预设优化方法
model = HDR()
#state_dict=torch.load('fashion_net_params.pth') 
#model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=mom) #随机梯度下降
stage2 = time.time()
print("Set Model:", stage2-start,'s')

#4. 训练
losses = []       #记录训练集损失
acces = []        #记录训练集准确率
eval_losses = []  #记录测试集损失
eval_acces = []  #记录测试集准确率
for epoch in range(num_epoches):
    prev = time.time()
    train_loss = 0
    train_acc = 0
    model.train()     
    if epoch%5==0:#动态修改参数学习率
        optimizer.param_groups[0]['lr'] *= 0.9
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

#5. 测试
    eval_loss = 0
    eval_acc = 0
    model.eval()

    for img, label in test_dataloader:
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
    
stage3 = time.time()
print("Training and Testing:", stage3-start, 's')

#6. 误差、正确率随时间变化图像
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
# plt.legend(handles=[l1,l2,l3,l4],labels=['Training Losses','Training Acces','Test Losses','Test Acces'], prop = {'size': 20}, loc='best')
# plt.xlabel("Training Time", fontdict={'size': 20})
# plt.ylabel("Value", fontdict={'size': 16})
# plt.title("Training Process", fontdict={'size': 30})
# plt.savefig('D:\E\File\学习\大二\机器学习/神经网络/训练过程Fashion.png',dpi=100)

#7. 保存模型参数
torch.save(model.state_dict(),'fashion_net_params.pth')
end = time.time()
print("End:", end-start, 's')

#6. 可视化
# for images,label in test_dataloader:
#     IMG = images.to(device)
#     out = model(IMG)
#     _, pred = out.max(1)
#     for i in range(label.shape[0]):
#         img = images[i]
#         if(pred[i] != label[i]):    
#             img = img.numpy()
#             img = np.transpose(img, (1, 2, 0))
#             window_name = class_list[int(pred[i])]+' '+class_list[int(label[i])]
#             cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#             cv2.resizeWindow(window_name, 500, 500)
#             cv2.imshow(window_name ,img)
#             cv2.waitKey(1000)
#             cv2.destroyAllWindows()