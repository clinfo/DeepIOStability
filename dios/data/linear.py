import numpy as np
import glob
import os
from numba import jit
import json
from dios.data.util import minmax_normalize, save_dataset


def f(x,u,A,B):
    return  A.dot(x.reshape(-1,1)) + B.dot(u)

def generate(N):
    m = 1
    d = 1
    k = 1

    A = np.array([[0,1],[-k/m,-d/m]])
    B = np.array([[1.0],[0]])
    C = np.array([[1.0,0]])

    dh = 1e-1
    T = 10
    times = np.arange(0,T,dh)
    step = times.shape[0]
    n =  A.shape[0]

    np.random.seed(0)


    x_data = np.zeros((N,times.shape[0],n))
    u_data = np.zeros((N,times.shape[0],1))
    y_data = np.zeros((N,times.shape[0],1))

    for i_N in range(N):

    # random binary input
        u=2 * np.random.randint(0,2,[step]) -1

        x = np.zeros((times.shape[0],n))

        #  初期値はランダム
        x0 = np.zeros(n)


        x[0] = x0

        for k in range(times.shape[0]-1):
            x[k+1] = x[k] + dh*f(x[k],u[k],A,B).reshape(-1)

        y = C.dot(x.T)

        u_data[i_N,:,0] = u
        x_data[i_N,:,:] = x
        y_data[i_N,:,0] = y

    y=np.array(y_data,dtype=np.float32)
    u=np.array(u_data,dtype=np.float32)
    x=np.array(x_data,dtype=np.float32)
    return x,u,y,y


def generate_dataset(N = 10000,M = 9000,name = "linear", path="dataset"):
    x_data, u_data, y_data, ys_data = generate(N)
    x_data, u_data, y_data, ys_data = minmax_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name)


