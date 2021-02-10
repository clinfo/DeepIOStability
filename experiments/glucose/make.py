import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from numba import jit
import json


#  Normal value

k_max = 0.0558
k_min = 0.0080
k_abs = 0.057
k_gri =0.0558
f = 0.9

b = 0.82
c = 0.010


BW = 78

D0 = BW * 1.0 * 1000

alpha = 0.00013
beta = 0.00236

D_alpha = 6.40e4
D_beta = 7.80e2
@jit
def k_empty(Qsto):
    return k_min + (k_max-k_min)/2 * (np.tanh(alpha*(Qsto - D_alpha)) - np.tanh(beta*(Qsto - D_beta))+2)

@jit
def fq(x,u):
    Qsto1 = x[0]
    Qsto2 = x[1]
    Qgut = x[2]
    Qsto = Qsto1 + Qsto2
    dQsto1 = - k_gri * Qsto1 +u[0]
    dQsto2 = - k_empty(Qsto) * Qsto2 + k_gri * Qsto1
    dQgut = - k_abs * Qgut + k_empty(Qsto) * Qsto2
    return np.array([dQsto1,dQsto2,dQgut])

dh = 1e-0
T = 300
times = np.arange(0,T,dh)
x = np.zeros((times.shape[0],3))

# 入力データの作成
dt_u = 30
u = np.zeros((times.shape[0],1))
u[times<=dt_u] = D0 * 1.0/dt_u

# 血中グルコース入力の計算
x0 = np.array([0,0,0])
x[0,:] = x0
for k in range(times.shape[0]-1):
    x[k+1] = x[k] + dh* fq(x[k],u[k])

Ra =  f * k_abs * (x[:,2])/BW

dh = 1e0
T = 1000
N = 10000
M = 9000
times = np.arange(0,T,dh)

np.random.seed(0)

u_data = np.zeros((N,times.shape[0],1))
x_data = np.zeros((N,times.shape[0],3))
y_data = np.zeros((N,times.shape[0],1))
ys_data = np.zeros((N,times.shape[0],1))


for i_sample in range(N):
    u0 = 1  + 0.1 *np.random.randn(3)# D0 + 0.1 D0 sigma [mg]
    u0[u0<0] = 0
    u_times = times[np.random.randint(0,times.shape[0],3)]

    u = np.zeros((times.shape[0],1))
    for i in range(3):
        u[(u_times[i]<=times) & (times<u_times[i] + dt_u)] = D0 * u0[i]/dt_u



    x = np.zeros((times.shape[0],3))

    for k in range(times.shape[0]-1):
        x[k+1] = x[k] + dh* fq(x[k],u[k])

    y =  f * k_abs * (x[:,2:3])/BW

    u_data[i_sample,:,:] = u
    x_data[i_sample,:,:] = x
    y_data[i_sample,:,:] = y
    ys_data[i_sample,:,:] = np.zeros(y.shape)


x_max=np.max(x_data[:,:,:])
y_max=np.max(y_data[:,:,:])
u_max=np.max(u_data[:,:,:])
x_data=x_data/x_max
y_data=y_data/y_max
u_data=u_data/u_max
max_data={"name":"glucose","x_max":x_max,"y_max":y_max,"u_max":u_max}
json.dump(max_data,open("dataset/max_data.json","w"))
print("[SAVE]","dataset/max_data.json")


os.makedirs("dataset",exist_ok=True)
filename="dataset/glucose.train.obs.npy"
print("[SAVE]",filename)
print(y_data[:M].shape)
np.save(filename,y_data[:M])
filename="dataset/glucose.test.obs.npy"
print("[SAVE]",filename)
print(y_data[M:].shape)
np.save(filename,y_data[M:])
filename="dataset/glucose.obs.npy"
print("[SAVE]",filename)
print(y_data.shape)
np.save(filename,y_data)
f
filename="dataset/glucose.train.input.npy"
print("[SAVE]",filename)
print(u_data[:M].shape)
np.save(filename,u_data[:M])
filename="dataset/glucose.test.input.npy"
print("[SAVE]",filename)
print(u_data[M:].shape)
np.save(filename,u_data[M:])

filename="dataset/glucose.train.state.npy"
print("[SAVE]",filename)
print(x_data[:M].shape)
np.save(filename,x_data[:M])
filename="dataset/glucose.test.state.npy"
print("[SAVE]",filename)
print(x_data[M:].shape)
np.save(filename,x_data[M:])

filename="dataset/glucose.train.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys_data[:M])
filename="dataset/glucose.test.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys_data[M:])

