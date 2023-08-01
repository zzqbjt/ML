import torch 
from torch import nn
from torch import optim
import numpy as np 
import matplotlib.pyplot as plt
from torch.nn import utils
from d2l import torch as d2l

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
torch.set_default_dtype(torch.float32)  

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

class Markov:
    def __init__(self, model, tau):
        super().__init__()
        self.func = model
        self.t = tau
    
    def predict(self, x):
        y = []
        for i in range(self.t):
            y.append(x[i])
        for i in range(self.t, len(x)):
            X = torch.tensor(x[i-self.t: i]).float().to(device)
            y.append(float(self.func(X)))
        return y
    
    def generate(self, init, length):
        y = []
        for i in range(len(init)):
            y.append(init[i])

        for i in range(self.t, length):
            X = torch.tensor(y[i-self.t: i]).float().to(device)
            y.append(float(self.func(X)))
        return y
    
    def train(self, y, epoches, lr, mom):
        y = torch.from_numpy(y)
        model = self.func.cuda()
        criterion = nn.MSELoss()        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)

        for epoch in range(epoches):

            # if epoch != 0 and epoch % 5 == 0:
            #     optimizer.param_groups[0]['lr'] *= 0.95

            train_loss = 0
            model.train()
            for i in range(self.t, len(y)):
                X = y[i-self.t: i].clone().detach().float().to(device)
                pred = model(X)
                pred = pred.to(device)

                x = torch.tensor([float(y[i])]).to(device)
                loss = criterion(pred, x)
                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step()

                train_loss += float(loss)
            
            print("epoch", epoch+1, ":", train_loss) 

        self.func = model

class RNN(nn.Module):
    def __init__(self, input, hidden, init = None):
        super().__init__()
        self.rnn = nn.RNN(input, hidden)
        self.linear  = nn.Linear(hidden, 1)
        self.num_hiddens = hidden
        if init:
            self.init = init.cuda()
        else:
            self.init = torch.rand((1, hidden)).cuda()
    def forward(self, input, state):
        Y, state= self.rnn(input, state)
        output = self.linear(Y)
        return output, state

def train(model, y, epoches, lr, mom):
        model = model.cuda()
        criterion = nn.MSELoss()        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
        state = model.init
        lr_max = lr
        lr_min = lr / 2
        for epoch in range(epoches):
            train_loss = 0
            state = model.init

            optimizer.param_groups[0]['lr'] = lr_min+(lr_max-lr_min)/2*(np.cos(epoch/epoches*np.pi)+1)
            
            for i in range(len(y)-1):
                state = state.detach()                     
                X = torch.tensor(y[i]).reshape(1, 1).float().to(device)
                label = torch.tensor(y[i+1]).reshape(1).float().to(device)

                pred, state = model(X, state)
                pred = pred.to(device).reshape(1)
                loss = criterion(pred, label)

                optimizer.zero_grad() 
                loss.backward()  
                utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                optimizer.step() 
                
                train_loss += float(loss)   
   
            print("epoch", epoch+1, ":", float(loss)) 

def generate(init, length, model):
    y = [init[0]]
    state = model.init
    for i in init[1:]: #预热warm up
        input = torch.tensor(y[-1]).to(device).reshape((1, 1)).float()
        _, state = model(input, state)
        y.append(i)
    for i in range(length-len(init)):
        input = torch.tensor(y[-1]).to(device).reshape((1, 1)).float()
        pred, state = model(input, state)
        y.append(float(pred.reshape(1)))
    return y


width = 20
step = 0.05
x = np.array(range(0, int(width/step))) * step
np.linspace
y = np.sin(x)
tau = 10

Model1 = Markov(MLP(tau, 1, [10, 10]).cuda(), tau)
Model1.train(y, 100, 0.001, 0.9)
print("Model1 is OK")

Model2 = RNN(1, 128)
train(Model2, y, 500, 0.001, 0.9)
print("Model2 is OK")

pred1 = Model1.generate(y[0:tau], len(y))
pred2 = generate(y[0:int(len(y)/2)], len(y), Model2)


plt.figure(figsize=(20, 20), dpi=50)
l1 = plt.scatter(x, y, c='red', s=100)
l2 = plt.scatter(x, pred1, c='blue')
l3 = plt.scatter(x, pred2, c='green')
plt.legend(handles=[l1, l2, l3], labels=['real', 'Marcov', 'RNN'], prop = {'size': 20}, loc='best')
plt.xlabel("x", fontdict={'size': 20})
plt.ylabel("y", fontdict={'size': 20})
plt.title("y-x", fontdict={'size': 30})
plt.savefig('y-x.png',dpi=100)
plt.show()



