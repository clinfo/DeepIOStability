import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from numba import jit

# read parameters
df_para = pd.read_csv("parameters/GImodel.csv")


np.random.seed(1)
b1 = df_para.b1
b2 = df_para.b2
b4 = df_para.b4
b5 = df_para.b5
b6 = df_para.b6
b7 = df_para.b7
G0 = df_para.Gb
I0 = df_para.Ib

k_gri =0.0558
D0 = 500000
# k_empt = (0.0558 + 0.0080)/2
k_abs = 0.057
f = 0.9
BW = 60

b = 0.82
k_max = 0.0558
k_min = 0.0080


@jit
def fq(x,u):
    Qsto1 = x[0]
    Qsto2 = x[1]
    Qgut = x[2]
    D = u[0]
    Qsto = Qsto1+Qsto2
    alpha = 5/ (2*D0*(1-b))
    k_empt = k_min + (k_max-k_min)/2 * (np.tanh(alpha*(Qsto - b * D0))+1)
    fq0 = - k_gri * Qsto1 + D
    fq1 = - k_empt * Qsto2 + k_gri * Qsto1
    fq2 = - k_abs * Qgut + k_empt * Qsto2
    return np.array([fq0,fq1,fq2])


N = 10000
dh = 1e0
T = 1000
times = np.arange(0,T,dh)

u_data = np.zeros((N,times.shape[0],1))
x_data = np.zeros((N,times.shape[0],6))
y_data = np.zeros((N,times.shape[0],2))



np.random.seed(0)
  
dt_u = 30 # 30[min]
i_sample = 2
i_Pt = 3
for i_Pt in range(b1.shape[0]):
    for i_sample in range(N):
        x = np.zeros((times.shape[0],3))
        u0 = 500000 + 100000 *np.random.randn(3)# 500[g]+- 250[g] sigma
        u0[u0<0] = 0
        u_times = times[np.random.randint(0,times.shape[0],3)]

        u = np.zeros((times.shape[0],1))
        for i in range(3):
            u[(u_times[i]<=times) & (times<u_times[i] + dt_u)] = u0[i]/dt_u

        x0 = np.array([0,0,0])
        x[0,:] = x0
        for k in range(times.shape[0]-1):
            x[k+1] = x[k] + dh* fq(x[k],u[k])

        h = f*k_abs*x[:,2]/BW/10

        G = np.zeros((times.shape[0]))
        I = np.zeros((times.shape[0]))
        X = np.zeros((times.shape[0]))


        ks = len(times[times < b5[i_Pt]])
        G[:ks] = G0[i_Pt]
        I[:ks] = I0[i_Pt]
        X[:ks] = G0[i_Pt] * b5[i_Pt]

        for k in range(ks-1,times.shape[0]-1,1):
            G[k+1] = G[k] + dh * (-b1[i_Pt] * G[k] - b4[i_Pt]*G[k]*I[k] + b7[i_Pt]+h[k])
            I[k+1] = I[k] + dh * (-b2[i_Pt] * I[k] + b6[i_Pt]/b5[i_Pt] *X[k])
            X[k+1] = X[k] + dh * (G[k] - G[k-ks])
            if G[k+1]<0:
                G[k+1] = 0
            if I[k+1]<0:
                I[k+1] = 0
            if X[k+1]<0:
                X[k+1] = 0

        y = np.c_[G,I]


        u_data[i_sample,:,:] = u
        x_data[i_sample,:,:] = np.c_[x,G,I,X]
        y_data[i_sample,:,:] = y
        
    print('## {0:02d}'.format(i_Pt))
    np.save('data/GlucoseInsulin{0:02d}_u_data.npy'.format(i_Pt),u_data)
    np.save('data/GlucoseInsulin{0:02d}_x_data.npy'.format(i_Pt),x_data)
    np.save('data/GlucoseInsulin{0:02d}_y_data.npy'.format(i_Pt),y_data)


