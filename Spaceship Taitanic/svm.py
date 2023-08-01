from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib

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
Flabel = np.array(Flabel)

train_data = Fdata[0:l, :]
train_label = Flabel[0:l]  

valid_data = Fdata[l:L, :]
valid_label = Flabel[l:L]

svc = svm.SVC(kernel='rbf', C=1.2, gamma='auto').fit(train_data, train_label)
joblib.dump(svc, path+ 'params_svm.pkl')