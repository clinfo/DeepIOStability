import numpy as np
import glob
import os
from numba import jit
import json
from dios.data.util import minmax_normalize, z_normalize, save_dataset


@jit
def f(x,u):
    return  x*(1- x**2) + u

def generate(N=20000):
    np.random.seed(0)
    dh = 1e-1
    T = 10
    times = np.arange(0,T,dh)

    # ステップ入力の設定
    """
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
    """
    n_steps=times.shape[0]
    u = np.zeros((N,n_steps,1))
    for i in range(N):
        r=np.random.randint(0,4)
        if r==0:
            s1=np.random.randint(1,n_steps)
            s2=np.random.randint(1,n_steps)
            ss=sorted([s1,s2])
            u[i,ss[0]:ss[1],0]=1
        elif r==1:
            s1=np.random.randint(1,n_steps)
            s2=np.random.randint(1,n_steps)
            ss=sorted([s1,s2])
            u[i,ss[0]:ss[1],0]=-1
        elif r==2:
            s1=np.random.randint(1,n_steps)
            s2=np.random.randint(1,n_steps)
            s3=np.random.randint(1,n_steps)
            s4=np.random.randint(1,n_steps)
            ss=sorted([s1,s2,s3,s4])
            u[i,ss[0]:ss[1],0]=1
            u[i,ss[2]:ss[3],0]=-1
        elif r==3:
            s1=np.random.randint(1,n_steps)
            s2=np.random.randint(1,n_steps)
            s3=np.random.randint(1,n_steps)
            s4=np.random.randint(1,n_steps)
            ss=sorted([s1,s2,s3,s4])
            u[i,ss[0]:ss[1],0]=-1
            u[i,ss[2]:ss[3],0]=1


    #  初期値は0
    """
    x01 = -np.ones(N//2)
    x02 = np.ones(N//2)
    x0 = np.concatenate([x01,x02],axis=0)
    np.random.shuffle(x0)
    """
    x0 = -np.ones(N)
    for i in range(N):
        r=np.random.randint(0,2)
        if r==0:
            x0[i]=1
        elif r==1:
            x0[i]=-1
    x = np.zeros((N,times.shape[0],1))

    x[:,0,0] = x0

    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])

    ys = np.ones(x.shape)
    ys[x<0] = -1
    return x,u,x,ys

def generate_dataset(N = 20000,M = 18000,name = "bistable", path="dataset"):
    x_data, u_data, y_data, ys_data = generate(N)
    x_data, u_data, y_data, ys_data = z_normalize(x_data, u_data, y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name)


