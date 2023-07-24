import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
import matplotlib
import time
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics

def sim(a, b):#求相似度，即交并比
    inter = a&b 
    union = a|b
    return len(inter)/len(union)

#颜色池
colorset = ['r', 'orange', 'gold', 'g', 'b', 'c', 'purple', 'black', 'gray', 'brown', 'y', 'darkgreen', 'cyan', 'darkblue', 'm', 'chocolate', 'deeppink', 'blueviolet', 'coral', 'olive', 'dodgerblue', 'orchid', 'peru', 'teal','r', 'orange', 'gold', 'g', 'b', 'c', 'purple']

t1 = time.time() #计时
#载入数据
datalist = np.loadtxt('D:\E\File\学习\大二\机器学习\聚类\TrajectoryData_students003\students003.txt')
#模型初始化
model = DBSCAN(eps=1.5, min_samples=3) 
#视频导出
fig =plt.figure(figsize = (10, 10), dpi = 100)
metadata = dict(title='Group Discovery', artist='zzn', comment='')
writer = FFMpegWriter(fps=12, metadata=metadata)
ffmpegpath = os.path.abspath("D:\D\Software/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

with writer.saving(fig, "Group Discovery.mp4", 100):
    plt.ion()
    i = 0
    for t in range(0, 5401, 10):
        xdata = []
        ydata = []
        pointlist = []
        IDlist = []

        #读取当前帧的点数据
        while i < len(datalist) and datalist[i][0] == t:
            xdata.append(datalist[i][2])
            ydata.append(datalist[i][3])
            pointlist.append([datalist[i][2], datalist[i][3]])
            IDlist.append(datalist[i][1])
            i += 1
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        pointlist = np.array(pointlist)
        yhat = model.fit_predict(pointlist) #运用DBSCAN分类

        #根据yhat将行人分到每一类中，列表的第i项为一个集合，表示第i簇。IDclusters存放每个行人的ID信息，pointclusters存放每个行人的坐标信息
        #colors存放每个簇的颜色
        colors = [] 
        IDclusters = [] 
        pointclusters = []
        scatters = yhat.max()+1

        for cls in range(yhat.max()+1):
            IDclusters.append(set())
            pointclusters.append([])
            for j in range(len(yhat)):
                if yhat[j] == cls:
                    IDclusters[cls].add(IDlist[j])
                    pointclusters[cls].append(pointlist[j])
        for j in range(len(yhat)):
            if yhat[j] == -1:
                IDclusters.append(set())
                pointclusters.append([])
                IDclusters[scatters].add(IDlist[j])
                pointclusters[scatters].append(pointlist[j])
                scatters += 1
        print(pointclusters)

        #初始帧按顺序赋颜色
        if t==0:
            prevID = [] #prevID和prevColor用于记录上一帧的聚类情况及每个类对应的颜色
            prevColor = []
            for j in range(len(IDclusters)):
                colors.append(j)
        else:
            for j in range(len(IDclusters)):
                colors.append(-1)        
        usedcolorset = set()
        for j in range(len(IDclusters)):
            for k in range(len(prevID)):
                if sim(IDclusters[j], prevID[k]) >= 0.8: #交并比≥0.8，则将该簇的颜色标记为上一帧中的颜色，并将该颜色标记为已占用，防止被重复使用
                    colors[j] = prevColor[k]
                    usedcolorset.add(prevColor[k])
        
        for j in range(len(colors)): #对于在上一帧中没有出现过的簇，从颜色池中按顺序选取没有被占用的颜色。
            if colors[j] == -1:
                color = 0
                while color in usedcolorset:
                    color += 1
                colors[j] = color
                usedcolorset.add(color)
        prevID = IDclusters #更新prevID和prevColor
        prevColor = colors

        plt.xlim(0, 15)
        plt.ylim(0, 15)

        for j in range(len(colors)): #对于每个簇单独画图
            p = np.array(pointclusters[j])
            xdata = p[:,0]
            ydata = p[:,1]
            plt.scatter(xdata, ydata, c=colorset[colors[j]])
        
        plt.show() #显示图像
        plt.pause(0.05)
#        writer.grab_frame()
        plt.clf()
    
    t2 = time.time()
    print("over:",t2-t1) #计时