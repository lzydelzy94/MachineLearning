# coding:utf-8
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')


def getTheta(x,y):
    theta = np.linalg.inv(x.T*x)*x.T*y
    return theta

path = "../Data/ex1data2.txt"
data = pd.read_csv(path,header=None, names=['Size','bedroom','price'])
data.insert(0,'Ones',1)

x = np.mat(data.iloc[:,:3].values)
y = np.mat(data.iloc[:,3:].values)

theta = getTheta(x,y)

x1= np.linspace(data.Size.min(),data.Size.max(),100)
x2= np.linspace(data.bedroom.min(),data.bedroom.max(),100)
x1,x2 = np.meshgrid(x1,x2)
f = theta[0,0]+theta[1,0]*x1+theta[2,0]*x2

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(x1,x2,f)
ax.scatter(data.Size,data.bedroom,data.price)

plt.show()

