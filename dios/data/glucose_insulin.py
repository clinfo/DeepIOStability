import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from numba import jit
import json
from dios.data.util import minmax_normalize, save_dataset
import  dios.data.glucose

def generate(N, Ra):
    # Sub4
    k1 = 3.35E-2
    k2 = 5.22E-5
    k3 = 1.055
    k4 = 0.293
    g0 = 3.13
    tau = 6

    #
    VG = 0.8

    # 平衡点の導出
    Gs =  (-k1*k3 + np.sqrt((k1*k3)**2 + 4*k2*k3*k4*g0))/(2*k2*k4)
    Is = (-k1*k3 + np.sqrt((k1*k3)**2 + 4*k2*k3*k4*g0))/(2*k2*k3)
    Xs = (-k1*k3 + np.sqrt((k1*k3)**2 + 4*k2*k3*k4*g0))/(2*k2*k4) *tau


    #Ra = np.load('../glucose/dataset/glucose.obs.npy')

    dh = 1e0
    T = Ra.shape[1]
    times = np.arange(0,T,dh)

    u_data = np.zeros((N,times.shape[0],1))
    x_data = np.zeros((N,times.shape[0],3))
    y_data = np.zeros((N,times.shape[0],2))
    ys_data = np.zeros((N,times.shape[0],2))


    for i_sample in range(N):

        G = Gs * np.ones((times.shape[0]))
        I = Is * np.ones((times.shape[0]))
        X = Xs * np.ones((times.shape[0]))
        u = Ra[i_sample]/VG
        
        for k in range(tau,times.shape[0]-1,1):
            G[k+1] = G[k] + dh * (-k1* G[k] - k2*G[k]*I[k] + g0+u[k])
            I[k+1] = I[k] + dh * (-k3 * I[k] + k4/tau *X[k])
            X[k+1] = X[k] + dh * (G[k] - G[k-tau])
        
        y = np.c_[G,I]
        x = np.c_[G,I,X]
        u_data[i_sample,:,:] = u
        x_data[i_sample,:,:] = x
        y_data[i_sample,:,:] = y
        ys_data[i_sample,:,:] = np.c_[Gs * np.ones((times.shape[0])),Is * np.ones((times.shape[0]))]
    return x_data, u_data, y_data, ys_data

def generate_dataset(N = 10000,M = 9000,name = "glucose_insulin", path="dataset"):
    _, _, y, _ =  dios.data.glucose.generate(N)
    x_data, u_data, y_data, ys_data = generate(N, Ra=y)
    x_data, u_data, y_data, ys_data = minmax_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name)


