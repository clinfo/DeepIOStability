import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os


def f(x,u):
    return  x*(1- x**2) + u
np.random.seed(0)
N = 20000
M = 18000
dh = 1e-1
T = 10
times = np.arange(0,T,dh)

# ステップ入力の設定

u1 = np.zeros((N//2,times.shape[0],1))
u_step = 4*np.random.rand(N//2)
tmp =  np.ones((1,times.shape[0])) *u_step.reshape(-1,1)
u1 [times<tmp,:]= 1

u_step = 4*np.random.rand(N//2)
tmp =  np.ones((1,times.shape[0])) *u_step.reshape(-1,1)
u1 [(times<tmp+5)&(5<=times),:]= -1

u2 = np.zeros((N//2,times.shape[0],1))
u_step = 4*np.random.rand(N//2)
tmp =  np.ones((1,times.shape[0])) *u_step.reshape(-1,1)
u2 [times<tmp,:]= -1

u_step = 4*np.random.rand(N//2)
tmp =  np.ones((1,times.shape[0])) *u_step.reshape(-1,1)
u2 [(times<tmp+5)&(5<=times),:]= 1


u=np.concatenate([u1,u2],axis=0)
np.random.shuffle(u)

#  初期値は0
x01 = -np.ones(N//2)
x02 = np.ones(N//2)
x0 = np.concatenate([x01,x02],axis=0)
np.random.shuffle(x0)

x = np.zeros((N,times.shape[0],1))

x[:,0,0] = x0

for k in range(times.shape[0]-1):
    x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])

ys = np.ones(x.shape)
ys[x<0] = -1

os.makedirs("data",exist_ok=True)

np.save('data/bistable_2step.train.state.npy',x[:M,:,:])
np.save('data/bistable_2step.train.obs.npy',  x[:M,:,:])
np.save('data/bistable_2step.train.input.npy',u[:M,:,:])
np.save('data/bistable_2step.train.stable.npy',ys[:M,:,:])

np.save('data/bistable_2step.test.state.npy',x[M:,:,:])
np.save('data/bistable_2step.test.obs.npy',  x[M:,:,:])
np.save('data/bistable_2step.test.input.npy',u[M:,:,:])
np.save('data/bistable_2step.test.stable.npy',ys[M:,:,:])

