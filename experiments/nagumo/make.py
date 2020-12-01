import numpy as np
import matplotlib.pyplot as plt
import os


def f(x,u):
    a = 0.7
    b = 0.8
    c = 3
    dx0 =  c*(x[:,0] + x[:,1]  - (x[:,0] **3)/3 + u[:,0])
    dx1 =  (-x[:,0] - b*x[:,1] + a)/c
    
    return  np.array([dx0,dx1]).T

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

np.random.seed(10)
n = 2
N = 10000
M = 9000
dh = 1e-1
T = 100

times = np.arange(0,T,dh)

Uon = -0.5
Uoff = 0

Toff =  0.5*T*np.random.rand(N)
Ton  =  0.5*T*np.random.rand(N)


x=get_stable_x()
xs =x[0,-1,:]
print(xs)
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

os.makedirs("dataset",exist_ok=True)
filename="dataset/nagumo.train.obs.npy"
print("[SAVE]",filename)
print(y[:M].shape)
np.save(filename,y[:M])
filename="dataset/nagumo.test.obs.npy"
print("[SAVE]",filename)
print(y[M:].shape)
np.save(filename,y[M:])

filename="dataset/nagumo.train.input.npy"
print("[SAVE]",filename)
print(u[:M].shape)
np.save(filename,u[:M])
filename="dataset/nagumo.test.input.npy"
print("[SAVE]",filename)
print(u[M:].shape)
np.save(filename,u[M:])

filename="dataset/nagumo.train.state.npy"
print("[SAVE]",filename)
print(x[:M].shape)
np.save(filename,x[:M])
filename="dataset/nagumo.test.state.npy"
print("[SAVE]",filename)
print(x[M:].shape)
np.save(filename,x[M:])

#filename="dataset/nagumo.train.stable.npy"
#print("[SAVE]",filename)
#np.save(filename,ys[:M])
#filename="dataset/nagumo.test.stable.npy"
#print("[SAVE]",filename)
#np.save(filename,ys[M:])

