import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os


def f(x,u):
    return  x*(1- x**2) + u
np.random.seed(0)
N = 10000
dh = 1e-1
T = 10
times = np.arange(0,T,dh)

u = 0.1*np.random.randn(N,times.shape[0],1)

#  初期値は0
x0 = np.zeros(N)
x = np.zeros((N,times.shape[0],1))

x[:,0,0] = x0

for k in range(times.shape[0]-1):
    x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])

ys = np.ones(x.shape)
ys[x<0] = -1

os.makedirs("data",exist_ok=True)

M=9000
np.save('data/bistable_zero.train.state.npy',x[:M,:,:])
np.save('data/bistable_zero.train.obs.npy',  x[:M,:,:])
np.save('data/bistable_zero.train.input.npy',u[:M,:,:])
np.save('data/bistable_zero.train.stable.npy',ys[:M,:,:])

np.save('data/bistable_zero.test.state.npy',x[M:,:,:])
np.save('data/bistable_zero.test.obs.npy',  x[M:,:,:])
np.save('data/bistable_zero.test.input.npy',u[M:,:,:])
np.save('data/bistable_zero.test.stable.npy',ys[M:,:,:])

