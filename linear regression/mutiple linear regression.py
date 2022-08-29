# coding:utf-8

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

def getCost(x,y,theta):
    inner = np.power(x*theta -y,2)
    return np.sum(inner)/(2*len(x))

def gradient(x,y,theta,alpha,times):
    costs = []
    for i in range(times):
        cost = getCost(x,y,theta)
        theta = theta - alpha*(x.T*(x*theta-y))/len(x)
        costs.append(cost)
    return theta,costs





path = "../Data/ex1data2.txt"
data = pd.read_csv(path,header=None,names=["Size","bedroom","price"])

#均值归一化
data = (data-data.mean())/(data.std())  #mean是平均数，std是标准差
data.insert(0,"Ones",1)

#提取数据
x = data.iloc[:,0:3].values
y = data.iloc[:,3:].values
theta = np.matrix([0,0,0]).T

#梯度下降
alpha = 0.01
times = 15000
thetaFinal,costs = gradient(x,y,theta,alpha,times)

#绘图
x1 = np.linspace(data.Size.min(),data.Size.max(),100)
x2= np.linspace(data.bedroom.min(),data.bedroom.max(),100)
x1,x2 = np.meshgrid(x1,x2)
f = thetaFinal[0,0]+thetaFinal[1,0]*x1+thetaFinal[2,0]*x2

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x1,x2,f,rstride=1,cstride=1)

ax.scatter(data.Size,data.bedroom,data.price,c='r')

plt.show()



