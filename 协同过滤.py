import numpy as np
import random
import math

def J(theta, X, Y, r, lam):
    return (np.sum((np.dot(X, theta.T)-Y)*r)**2)/2+lam/2*(np.sum(X*X)+np.sum(theta*theta))

m = 20
n = 20
Y = np.zeros((m, n))
r = np.ones((m, n))
TY = np.zeros((m, n))
Tr = np.ones((m, n))
for i in range(m):
    for j in range(n):
        Y[i][j] = random.randint(1, 10)
        TY[i][j] = Y[i][j]
        if Y[i][j] > 5:
            Y[i][j] = 0
            TY[i][j] = Y[i][j]
            r[i][j] = 0
            Tr[i][j] = r[i][j]
        else:
            x = random.randint(1, 10)
            if x == 1:
                Y[i][j] = 0
                r[i][j] = 0

l = 10
a = 0.01
lam = 0.0
for i in range(10):
    lam = i / 10
    epoches = 50
    Theta = np.random.random((n, l))
    X = np.random.random((m, l))
    for epoch in range(epoches):
        Theta = Theta - a * (np.dot(((np.dot(X, Theta.T)-Y)*r).T, X) + lam * Theta.sum())
        X = X - a * (np.dot(((np.dot(X, Theta.T)-Y)*r), Theta) + lam * X.sum())
        if epoch == epoches - 1:
            print("epoch: {}, train loss = {:.4f}, test loss = {:.4f}".format(epoch, J(Theta, X, Y, r, lam), J(Theta, X, TY, Tr, lam))) 

PRED = np.dot(X, Theta.T)
for i in range(m):
    for j in range(n):
        PRED[i][j] = round(PRED[i][j])
# print(PRED)
# print(Y)
# print(TY)