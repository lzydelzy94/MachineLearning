# coding:utf-8
import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import scipy.io as sio

def sigmoid(z):
    return 1/(1+np.exp(-z))

# def getCost(theta,x,y,alpha,lamda):
