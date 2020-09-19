import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
n = 2
np.random.seed(0)
A = np.array([[-0.5,0.1],[0.1,-0.3]])
B = np.array([[1.0],[0]])
C = np.array([[1.0,1.0]])

def f(x,u):
    return  A.dot(x.reshape(-1,1)) + B.dot(u)
N = 10000
dh = 1e-1
T = 10
times = np.arange(0,T,dh)
step = times.shape[0]



x_data = np.zeros((N,times.shape[0],n))
u_data = np.zeros((N,times.shape[0],1))
y_data = np.zeros((N,times.shape[0],1))

for i_N in range(N):
    input_t=np.random.randint(0,step-1,10)
    u=np.zeros((step))
    for t in input_t:
        u[t]=1.0

    x = np.zeros((times.shape[0],n))

    #  初期値はランダム
    x0 = np.zeros(n)


    x[0] = x0

    for k in range(times.shape[0]-1):
        x[k+1] = x[k] + dh*f(x[k],u[k]).reshape(-1)

    y = C.dot(x.T)

    u_data[i_N,:,0] = u
    x_data[i_N,:,:] = x
    y_data[i_N,:,0] = y

data_y=np.array(y_data,dtype=np.float32)
data_u=np.array(u_data,dtype=np.float32)
data_x=np.array(x_data_x,dtype=np.float32)
filename="dataset/sample.train.obs.npy"
print("[SAVE]",filename)
print(data_y.shape)
np.save(filename,data_y)
#np.save("dataset/sample.state.npy",x)
filename="dataset/sample.train.input.npy"
print("[SAVE]",filename)
print(data_u.shape)
np.save(filename,data_u)
filename="dataset/sample.train.state.npy"
print("[SAVE]",filename)
print(data_x.shape)
np.save(filename,data_x)
filename="dataset/sample.train.stable.npy"
print("[SAVE]",filename)
np.save(filename,np.zeros(y_data.shape,dtype=np.float32))
