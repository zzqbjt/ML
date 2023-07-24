import cv2    
import matplotlib.pyplot as plt  
import os
import os.path as osp
from skimage import io
import numpy as np
import joblib
from sklearn import metrics
from sklearn.svm import SVC

#计算hog特征函数
def hog_descriptor(image):
    if (image.max()-image.min()) != 0:
        image = (image - image.min()) / (image.max() - image.min())
        image *= 255
        image = image.astype(np.uint8)
    hog = cv2.HOGDescriptor()
    hog_feature = hog.compute(image)
    return hog_feature

#计算MR和FPR
def cal_MR(threshold, y, label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if y[i]>=threshold and label[i]==1:
            TP += 1
        elif y[i]<threshold and label[i]==0:
            TN += 1
        elif y[i]>=threshold and label[i]==1:
            FP += 1
        else:
            FN +=1
    MR = FN/(TP+FN)
    return MR

def cal_FPR(threshold, y, label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if y[i]>=threshold and label[i]==1:
            TP += 1
        elif y[i]<threshold and label[i]==0:
            TN += 1
        elif y[i]>=threshold and label[i]==0:
            FP += 1
        else:
            FN +=1
    FPR = FP/(TN+FP)
    return FPR
            
#读入图像
poslist = os.listdir('D:\E\File\学习\大二\机器学习\INRIAPerson\Train\pos')
neglist = os.listdir('D:\E\File\学习\大二\机器学习\INRIAPerson\Train/neg')
testlist = os.listdir('D:\E\File\学习\大二\机器学习\INRIAPerson\Test\pos')
testnlist = os.listdir('D:\E\File\学习\大二\机器学习\INRIAPerson\Test/neg')

#图像预处理
hog_list = []
label_list = []
for i in range(len(poslist)):
    posimg = io.imread(osp.join('D:\E\File\学习\大二\机器学习\INRIAPerson\Train\pos',poslist[i]))
    posimg = cv2.cvtColor(posimg,cv2.COLOR_RGBA2GRAY) #转为灰度图
    posimg = cv2.resize(posimg, (64, 128), interpolation=cv2.INTER_NEAREST) #转为64*128
    pos_hog = hog_descriptor(posimg) #计算hog特征
    hog_list.append(pos_hog) 
    label_list.append(1) #标注为1
for i in range(len(neglist)):
    negimg = io.imread(osp.join('D:\E\File\学习\大二\机器学习\INRIAPerson\Train/neg',neglist[i]))
    negimg = cv2.cvtColor(negimg,cv2.COLOR_RGBA2GRAY)
    negimg = cv2.resize(negimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    neg_hog = hog_descriptor(negimg)
    hog_list.append(neg_hog)
    label_list.append(0)
hog_list = np.float32(hog_list)
label_list = np.int32(label_list).reshape(len(label_list),1)

#训练SVM
clf = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
clf.fit(hog_list.squeeze(), label_list.squeeze())
joblib.dump(clf, "D:\E\File\学习\大二\机器学习/SVM/trained_svm.m")

#在测试集上验证
test_hog = []
test_label = []
for i in range(len(testlist)):
    testimg = io.imread(osp.join('D:\E\File\学习\大二\机器学习\INRIAPerson\Test\pos', testlist[i]))
    testimg = cv2.cvtColor(testimg, cv2.COLOR_RGBA2GRAY)
    testimg = cv2.resize(testimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    testhog = hog_descriptor(testimg)
    test_hog.append(testhog)
    test_label.append(1)
for i in range(len(testlist)):
    testnimg = io.imread(osp.join('D:\E\File\学习\大二\机器学习\INRIAPerson\Test/neg', testnlist[i]))
    tesnimg = cv2.cvtColor(testnimg, cv2.COLOR_RGBA2GRAY)
    testnimg = cv2.resize(testnimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    testnhog = hog_descriptor(testnimg)
    test_hog.append(testnhog)
    test_label.append(0)
test_hog = np.float32(test_hog)
test_label = np.int32(test_label).reshape(len(test_label),1)

clf = joblib.load("D:\E\File\学习\大二\机器学习/SVM/trained_svm.m")
prob = clf.predict_proba(test_hog.squeeze())[:,1]
precision, recall, thresholds = metrics.precision_recall_curve(test_label.squeeze(), prob)
fpr, tpr, thresholds = metrics.roc_curve(test_label.squeeze(), prob)
#计算MR vs FPR
MR = []
FPR = []
for threshold in range(0, 501): #取501个数据点
    MR.append(cal_MR(threshold/500, prob, test_label))
    FPR.append(cal_FPR(threshold/500, prob, test_label))
MR = np.array(MR)
FPR = np.array(FPR)

#绘制PR曲线
plt.figure(figsize=(20, 20), dpi=100)
plt.plot(precision, recall, c='red')
plt.xlabel("P", fontdict={'size': 16})
plt.ylabel("R", fontdict={'size': 16})
plt.title("PR", fontdict={'size': 20})
plt.savefig('D:\E\File\学习\大二\机器学习/SVM/PR.png',dpi=300)

#绘制ROC曲线
plt.figure(figsize=(20, 20), dpi=100)
plt.plot(fpr, tpr, c='red')
plt.xlabel("FPR", fontdict={'size': 16})
plt.ylabel("TPR", fontdict={'size': 16})
plt.title("ROC", fontdict={'size': 20})
plt.savefig('D:\E\File\学习\大二\机器学习/SVM/ROC.png',dpi=300)

#绘制MR vs FPR曲线
plt.figure(figsize=(20, 20), dpi=100)
plt.plot(FPR, MR, c='red')
plt.xlabel("FPR", fontdict={'size': 16})
plt.ylabel("MR", fontdict={'size': 16})
plt.title("MR vs FPR", fontdict={'size': 20})
plt.savefig('D:\E\File\学习\大二\机器学习/SVM/MRvsFPR.png',dpi=300)

#计算AUC
AUC = metrics.roc_auc_score(test_label.squeeze(), prob)
print(AUC)

#给图像画框
for i in range(len(testlist)):
    image = io.imread(osp.join('D:\E\File\学习\大二\机器学习\INRIAPerson\Test\pos',testlist[i]))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(image)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite('D:\E\File\Program\Python/result/'+testlist[i]+".jpg",image)
#    print(i)
print('over')




