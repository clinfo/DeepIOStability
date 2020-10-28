
import numpy as np
import os
np.random.seed(1234)
A=np.array([[-0.5,0.1],[0.1,-0.3]])
B=np.array([[1],[0]])
C=np.array([[0.4,0.6],[0.8,2]])
step=100
dt=0.1
N=1000
data_y=[]
data_u=[]
data_x=[]
for i in range(N): 
    x0=np.random.normal(0.0,0.5,(2,1))
    x=np.zeros((step,2,1))
    u=np.zeros((step,1))
    y=np.zeros((step,2,1))

    input_t=np.random.randint(0,step-1,10)
    for t in input_t:
      u[t,0]=1.0

    xt=x0
    x[0:,:]=x0[:,:]
    for t in range(step-1):
        x[t+1]=xt+dt*(np.dot(A,xt)+B*u[t,0])
        y[t]=np.dot(C,xt)
        xt=x[t+1]
    y[step-1]=np.dot(C,xt)
    data_y.append(y[:,:,0])
    data_x.append(x[:,:,0])
    data_u.append(u)
data_y=np.array(data_y,dtype=np.float32)
data_u=np.array(data_u,dtype=np.float32)
data_x=np.array(data_x,dtype=np.float32)
os.makedirs("dataset",exist_ok=True)
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

