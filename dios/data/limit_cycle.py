import numpy as np
import glob
import os
from numba import jit
import json
from dios.data.util import minmax_normalize, save_dataset


@jit
def f(x,u):
    dx0 =  x[:,0] - x[:,1]  - x[:,0] * (x[:,0]**2 +x[:,1]**2) + x[:,0] * u[:,0]
    dx1 =  x[:,0] + x[:,1]  - x[:,1] * (x[:,0]**2 +x[:,1]**2) +x[:,1] * u[:,0]
    return  np.stack((dx0,dx1)).T

def generate(N):
    np.random.seed(0)
    u_sigma = 0.5

    n = 2
    dh = 1e-1
    T = 10
    times = np.arange(0,T,dh)


    x0= np.random.randn(N,n)

    u = u_sigma * np.random.randn(N,times.shape[0],1)
    x = np.zeros((N,times.shape[0],x0.shape[1]))

    x[:,0,:] = x0

    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])


    ys = np.zeros((N,times.shape[0],x0.shape[1]))
    ys[:,:,0] = x[:,:,0]  / np.sqrt(x[:,:,0]**2 + x[:,:,1]**2)
    ys[:,:,1] = x[:,:,1]  / np.sqrt(x[:,:,0]**2 + x[:,:,1]**2)

    u=np.array(u,dtype=np.float32)
    x=np.array(x,dtype=np.float32)
    y=x
    ys=np.array(ys,dtype=np.float32)
    
    return x, u, y, ys

def generate_dataset(N = 10000,M = 9000,name = "limit_cycle", path="dataset"):
    x_data, u_data, y_data, ys_data = generate(N)
    x_data, u_data, y_data, ys_data = minmax_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name)


