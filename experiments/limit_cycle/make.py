import numpy as np
import matplotlib.pyplot as plt
import os

def f(x,u):
    dx0 =  x[:,0] - x[:,1]  - x[:,0] * (x[:,0]**2 +x[:,1]**2) + x[:,0] * u[:,0]
    dx1 =  x[:,0] + x[:,1]  - x[:,1] * (x[:,0]**2 +x[:,1]**2) +x[:,1] * u[:,0]
    return  np.array([dx0,dx1]).T


np.random.seed(0)
u_sigma = 0.5

N = 10000
M = 9000
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

os.makedirs("dataset",exist_ok=True)
filename="dataset/limit_cycle.train.obs.npy"
print("[SAVE]",filename)
print(y[:M].shape)
np.save(filename,y[:M])
filename="dataset/limit_cycle.test.obs.npy"
print("[SAVE]",filename)
print(y[M:].shape)
np.save(filename,y[M:])

filename="dataset/limit_cycle.train.input.npy"
print("[SAVE]",filename)
print(u[:M].shape)
np.save(filename,u[:M])
filename="dataset/limit_cycle.test.input.npy"
print("[SAVE]",filename)
print(u[M:].shape)
np.save(filename,u[M:])

filename="dataset/limit_cycle.train.state.npy"
print("[SAVE]",filename)
print(x[:M].shape)
np.save(filename,x[:M])
filename="dataset/limit_cycle.test.state.npy"
print("[SAVE]",filename)
print(x[M:].shape)
np.save(filename,x[M:])

filename="dataset/limit_cycle.train.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys[:M])
filename="dataset/limit_cycle.test.stable.npy"
print("[SAVE]",filename)
np.save(filename,ys[M:])

