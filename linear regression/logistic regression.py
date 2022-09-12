# coding:utf-8
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def getCost(theta,x,y):
    m = len(x)
    theta = np.mat(theta)
    h = sigmoid(x*theta)
    cost = np.sum((np.multiply(-y,np.log(h))-np.multiply((1-y),np.log(1-h)))/m)
    return cost

def grad(theta,x,y,alpha,times):
    m = len(x)
    costs = np.zeros((times,1))
    for i in range(times):
        cost = getCost(theta,x,y)
        costs[i] = cost
        h = sigmoid(x*theta)
        error = h-y
        theta = theta - alpha*np.dot(x.T,error)/m
    return theta,costs

def predict(theta,x):
    h = sigmoid(x * theta)
    return [1 if x >=0.5 else 0 for x in h]

def accuracy(theta,x,y):  #准确率
    probability = predict(theta,x)
    length = len(probability)
    sum = 0
    for i in range(length):
        if (probability[i] == y[i,0]):
            sum = sum+1
    return (sum*100)/length


path = "../Data/ex2/ex2data1.txt"
data = pd.read_csv(path,names=['x1','x2','y'])
#data.insert(0,'Ones',1)


#提取x和y,转化为矩阵，并初始化theta
x = data.iloc[:,:2]
y = np.mat(data.iloc[:,2:])
theta = np.matrix([0,0,0]).T

#对x进行均值标准化

x = (x-np.mean(x))/(np.max(x)-np.min(x))
x.insert(0,'Ones',1)
normal_x = np.mat(x)

thetaFinal,costs = grad(theta,normal_x,y,0.1,5000)

print(accuracy(thetaFinal,normal_x,y))


#可视化数据

#cost函数下降
fig,ax = plt.subplots(2,1)
ax[0].plot(costs)

#散点图

x.insert(3,'y',y)

#处理过后的特征值根据y值区分数据集
positive = x[x['y'].isin([1])]
negative = x[x['y'].isin([0])]

ax[1].scatter(positive.x1,positive.x2,marker='o',c='b')
ax[1].scatter(negative.x1,negative.x2,marker='x',c='r')

x_1 = np.linspace(x.x1.min(),x.x2.max(),100)
x_2 = (-thetaFinal[0,0]-thetaFinal[1,0]*x_1)/thetaFinal[2,0]
ax[1].plot(x_1,x_2)

plt.show()