import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getCost(x,y,theta):
    inner = np.power(x*theta-y,2)
    return np.sum(inner)/(2*len(x))

def gradient(x,y,theta,alpha,times):
    costs = []  #记录cost变化
    for i in range(times):
        error = x*theta - y
        theta = theta - alpha*(x.T*error/len(x))
        cost = getCost(x,y,theta)
        costs.append(cost)
    return theta,costs




#读取参数，加上列名
path = "../Data/ex1data1.txt"
data = pd.read_csv(path,sep=',',header=None,names=['Population','Profit'])
data.insert(0,'Ones',1)


x = np.matrix(data.iloc[:,:-1])
y = np.matrix(data.iloc[:,-1:])
theta = np.matrix(np.zeros(x.shape[1])).T

alpha = 0.01
times = 5000

theta,costs = gradient(x,y,theta,alpha,times)

#图形展示
# plt.scatter(data.Population,data.Profit,label='Data')
# plt.plot(x[:,1],x.dot(theta),color='r',label="Predication")
# plt.xlabel("population of city in 10,000")
# plt.ylabel("profit in dollar 10,000")
# plt.show()


fig,ax = plt.subplots(2,1)

#展示cost函数下降
ax[0].plot(costs)



ax[1].scatter(data.Population,data.Profit,label='Data')
ax[1].plot(x[:,1],x.dot(theta),color='r',label="Predication")
plt.show()


print(theta)

