# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def getCost(theta,x,y,alpha,lamda):
    theta = np.matrix(theta).T
    x = np.matrix(x)
    y = np.matrix(y)
    m = len(x)
    h = sigmoid(np.dot(x,theta))
    cost = np.sum((np.multiply(-y,np.log(h))-np.multiply((1-y),np.log(1-h)))/m)+np.sum(np.square(theta)*lamda/(2*m))
    return cost

def gradient(theta, x, y, alpha, lamda):
        theta = np.matrix(theta).T
        x = np.matrix(x)
        y = np.matrix(y)
        m = len(x)
        theta = np.mat(theta)

        h = sigmoid(x * theta)
        grad = alpha * np.dot(x.T, h - y) / m + theta * (alpha * lamda) / m  # fmit_tnc传递函数需要，这里直接用赋值alpha为0.1
        # grad = theta*(1-0.1*lamda/m) - 0.1*np.dot(x.T,h-y)/m
        return grad


def feature_mapping(x1, x2, degree):  # 特征映射
    dataTemp = {}
    for i in range(degree + 1):
        for j in range(i + 1):
            dataTemp['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(dataTemp)


path = "../Data/ex2/ex2data2.txt"
data = pd.read_csv(path,header=None,names=['x1','x2','y'])

degree = 6  #特征映射层数
x1 = data['x1']
x2 = data['x2']

data2 = feature_mapping(x1,x2,degree)

x = np.mat(data2)
y = np.mat(data.iloc[:,2:])
theta = np.matrix(np.zeros(28)).T

#不采用现成工具库，而通过自己迭代实现时，可能因为迭代次数不够，学习率过小等原因，导致正则参数为0时
#无法体现出过拟合的特点，  可通过增加迭代次数或者增大学习率来体现(增加迭代次数效果更明显)

#times = 5000
alpha = 200
lamda = 0
result = opt.fmin_tnc(func=getCost,x0=theta,fprime=gradient,args=(x,y,alpha,lamda))

thetaFinal = np.mat(result[0]).T

fig,ax = plt.subplots(figsize=(10,10))

#绘制散点
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

ax.scatter(positive.x1,positive.x2,marker='o',c='b')
ax.scatter(negative.x1,negative.x2,marker='x',c='r')

#绘制边界
x_1 = np.linspace(-1,1.5,250)
xx,yy = np.meshgrid(x_1,x_1)
z = feature_mapping(xx.ravel(),yy.ravel(),degree).values
z = z*thetaFinal
z = z.reshape(xx.shape)
plt.contour(xx,yy,z,0)
plt.show()


