import numpy as np
from numba import jit
import json
import glob
import os
from dios.data.util import minmax_normalize, save_dataset


@jit
def f(x,u):
    a = 0.7
    b = 0.8
    c = 3
    dx0 =  c*(x[:,0] + x[:,1]  - (x[:,0] **3)/3 + u[:,0])
    dx1 =  (-x[:,0] - b*x[:,1] + a)/c
    
    return  np.array([dx0,dx1]).T

@jit
def get_stable_x():
    n = 2
    N = 2 
    dh = 1e-2
    T = 100

    times = np.arange(0,T,dh)


    x0= np.zeros((N,n))
     
    x = np.zeros((N,times.shape[0],x0.shape[1]))

    x[:,0,:] = x0

    u = np.zeros((N,times.shape[0],1))
    u[1,:,:] = - 0.5

    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])    
    return x

def generate(N):
    np.random.seed(10)
    n = 2
    dh = 1e-1
    T = 100

    times = np.arange(0,T,dh)

    Uon = -0.5
    Uoff = 0

    Toff =  0.5*T*np.random.rand(N)
    Ton  =  0.5*T*np.random.rand(N)


    x=get_stable_x()
    xs =x[0,-1,:]
    x0= xs * np.ones((N,n)) 
    u =  Uoff *np.ones((N,times.shape[0],1))
    for i in range(N):
        j=0
        while(j*(Ton[i]+Toff[i])<=T):
            j = j+1
            u[i,((j-1)*(Toff[i]+Ton[i]) + Toff[i]<times)&(times <= j*(Toff[i]+Ton[i]))] =  Uon



    x = np.zeros((N,times.shape[0],x0.shape[1]))
    x[:,0,:] = x0
    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])




    u=np.array(u,dtype=np.float32)
    x=np.array(x,dtype=np.float32)
    y=x
    #ys=np.array(ys,dtype=np.float32)
    return x, u, y, None

def generate_dataset(N = 10000,M = 9000,name = "nagumo", path="dataset"):
    x_data, u_data, y_data, ys_data = generate(N)
    x_data, u_data, y_data, ys_data = minmax_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name)


