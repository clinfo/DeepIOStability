import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


beta = 8.5e-6
V0 = 1
delta = 0.02
C =  0.6e-15
R = 10e15
r = 60e3
Vcc = 5
Cl = 20e-15
Rl = 4.5e5

gamma0 = 1/(C+Cl)
gamma1 = 1/(C+Cl) + 1/C
gamma2 = 1/R + 1/r
gamma3 = 1/Rl + 1/r
gamma4 = 1/r

def I(Ud,Ug):
  if Ug <= V0:
    return 0
  elif Ug <= V0 + Ud:
    return beta*((Ug - V0 )**2) * (1 + delta * Ud)
  else :
    return beta* Ud*( 2*(Ug - V0) - Ud)* (1 + delta * Ud)


def f(x):
  f0 = gamma0 * (gamma4*x[1] + Vcc/ Rl - gamma3* x[0] - I(x[2],x[0]) + gamma4*x[2] - gamma2*x[3])
  f1 = gamma1 * (gamma4*x[0] - gamma2*x[1] - I(x[0],x[2])) + gamma0 * (gamma4*x[3] + Vcc/Rl - gamma3* x[2])
  f2 = gamma0 * (- I(x[0],x[2]) + gamma4*x[0] - gamma2* x[1] + gamma4*x[3] + Vcc/ Rl - gamma3*x[2])
  f3 = gamma0 * (gamma4*x[1] + Vcc/Rl - gamma3* x[0]) + gamma1 * ( - I(x[2],x[0]) + gamma4*x[2] - gamma2*x[3])
  return np.array([f0,f1,f2,f3])

B2 = np.array([0,gamma0,gamma0,0])
B1 = np.array([gamma0,0,0,gamma0])

N = 10000
h = 1e-11
T = 1e-8
times = np.arange(0,T,h)

N_step = 2
step_size =  1e-9
I_step = 3e-4
I_sigma = 0



x_data = np.zeros((N,times.shape[0],4))
y_data = np.zeros((N,times.shape[0],1))
u_data = np.zeros((N,times.shape[0],1))

start = time.time()
np.random.seed(1111)
for i_sample in range(N):
    x0 = Vcc *np.random.rand(4)
    i_step_p = np.random.randint(0,times.shape[0],N_step)
    i_step_n = np.random.randint(0,times.shape[0],N_step)

    u_step_p = np.zeros(times.shape)
    u_step_n = np.zeros(times.shape)

    for i in range(N_step):
        u_step_p[(times>= times[i_step_p[i]]- step_size)&(times<times[i_step_p[i]])]= I_step
        u_step_n[(times>= times[i_step_n[i]]- step_size)&(times<times[i_step_n[i]])]= - I_step

    u =  I_sigma*np.random.randn(times.shape[0]) +  u_step_p + u_step_n


    x = np.zeros((times.shape[0],4))
    x[0,:] = x0
    for k in range(times.shape[0]-1):
        x[k+1] = x[k] + h* (f(x[k]) +  B1*u[k])

    y = x[:,3]

    x_data[i_sample,:,:] = x
    u_data[i_sample,:,0] = u
    y_data[i_sample,:,0] = y
import os
os.makedirs("data",exist_ok=True)
"""
np.save('data/FlipFlop_x_data.npy',x_data)
np.save('data/FlipFlop_u_data.npy',u_data)
np.save('data/FlipFlop_y_data.npy',y_data)
"""
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print(x_data.shape)
print(y_data.shape)
print(u_data.shape)
y_data=np.array(y_data,dtype=np.float32)
u_data=np.array(u_data,dtype=np.float32)
x_data=np.array(x_data,dtype=np.float32)

M=9000
np.save('data/flipflop.train.state.npy',x_data[:M,:,:])
np.save('data/flipflop.train.obs.npy',y_data[:M,:,:])
np.save('data/flipflop.train.input.npy',u_data[:M,:,:])

np.save('data/flipflop.test.state.npy',x_data[M:,:,:])
np.save('data/flipflop.test.obs.npy',y_data[M:,:,:])
np.save('data/flipflop.test.input.npy',u_data[M:,:,:])

print(x_data.shape)
print(y_data.shape)
print(u_data.shape)
y_data=np.array(y_data,dtype=np.float32)
u_data=np.array(u_data,dtype=np.float32)
x_data=np.array(x_data,dtype=np.float32)

M=9000
np.save('data/flipflop.train.state.npy',x_data[:M,:,:])
np.save('data/flipflop.train.obs.npy',y_data[:M,:,:])
np.save('data/flipflop.train.input.npy',u_data[:M,:,:])

np.save('data/flipflop.test.state.npy',x_data[M:,:,:])
np.save('data/flipflop.test.obs.npy',y_data[M:,:,:])
np.save('data/flipflop.test.input.npy',u_data[M:,:,:])

