import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from numba import jit
import json


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


Ra = np.load('../glucose/dataset/glucose.obs.npy')

dh = 1e0
T = Ra.shape[1]
times = np.arange(0,T,dh)

N = 10000
M = 9000

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


os.makedirs("dataset",exist_ok=True)

x_max=np.max(x_data[:,:,:])
y_max=np.max(y_data[:,:,:])
u_max=np.max(u_data[:,:,:])
x_min=np.min(x_data[:,:,:])
y_min=np.min(y_data[:,:,:])
u_min=np.min(u_data[:,:,:])
x_data=(x_data-x_min)/(x_max-x_min)
y_data=(y_data-y_min)/(y_max-y_min)
u_data=(u_data-u_min)/(u_max-u_min)
minmax_data={"name":"glucose_insulin",
        "x_max":x_max,"y_max":y_max,"u_max":u_max,
        "x_min":x_min,"y_min":y_min,"u_min":u_min,
        }
filename="dataset/minmax_data.json"
json.dump(minmax_data,open(filename,"w"))
print("[SAVE]",filename)


filename="dataset/glucose_insulin.train.obs.npy"
print("[SAVE]",filename)
print(y_data[:M].shape)
np.save(filename,y_data[:M])
filename="dataset/glucose_insulin.test.obs.npy"
print("[SAVE]",filename)
print(y_data[M:].shape)
np.save(filename,y_data[M:])

filename="dataset/glucose_insulin.train.input.npy"
print("[SAVE]",filename)
print(u_data[:M].shape)
np.save(filename,u_data[:M])
filename="dataset/glucose_insulin.test.input.npy"
print("[SAVE]",filename)
print(u_data[M:].shape)
np.save(filename,u_data[M:])

filename="dataset/glucose_insulin.train.state.npy"
print("[SAVE]",filename)
print(x_data[:M].shape)
np.save(filename,x_data[:M])
filename="dataset/glucose_insulin.test.state.npy"
print("[SAVE]",filename)
print(x_data[M:].shape)
np.save(filename,x_data[M:])

filename="dataset/glucose_insulin.train.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys_data[:M])
filename="dataset/glucose_insulin.test.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys_data[M:])

