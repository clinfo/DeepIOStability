import numpy as np
import glob
import os
#GlucoseInsulin00_u_data.npy
def check(param_index):
    filename="data_org/GlucoseInsulin{:02d}_u_data.npy".format(param_index)
    u=np.load(filename)
    filename="data_org/GlucoseInsulin{:02d}_x_data.npy".format(param_index)
    x=np.load(filename)
    filename="data_org/GlucoseInsulin{:02d}_y_data.npy".format(param_index)
    y=np.load(filename)
    filename="data_org/GlucoseInsulin{:02d}_ys_data.npy".format(param_index)
    ys=np.load(filename)
    
    d=0
    print("d=",d)
    x_max=np.max(x[:,:,d])
    x_min=np.min(x[:,:,d])
    x0=x[:5,0,d]
    print("min x:",x_min)
    print("max x:",x_max)
    print("x0:",x0,"...")
     
    d=1
    print("d=",d)
    x_max=np.max(x[:,:,d])
    x_min=np.min(x[:,:,d])
    x0=x[:5,0,d]
    print("min x:",x_min)
    print("max x:",x_max)
    print("x0:",x0,"...")
    

    d=1
    print("d=",d)
    y_max=np.max(y[:,:,d])
    y_min=np.min(y[:,:,d])
    y0=y[:5,0,d]
    ys0=ys[:5,0,d]
    print("min y:",y_min)
    print("max y:",y_max)
    print("y0:",y0,"...")
    print("stable y:",ys0,"...")

    d=0
    print("d=",d)
    u_max=np.max(u[:,:,d])
    u_min=np.min(u[:,:,d])
    u0=u[:5,0,d]
    print("min u:",u_min)
    print("max u:",u_max)
    print("u0:",u0,"...")
    def compute_norm(yy,u):
        gy=np.sum(yy**2,axis=2)
        gu=np.sum(u**2,axis=2)

        gy=np.mean(gy,axis=1)
        gy=np.mean(gy,axis=0)
        gu=np.mean(gu,axis=1)
        gu=np.mean(gu,axis=0)
        return gy,gu

    yy=(y-ys)/y_max
    uu=u/u_max
    gy,gu=compute_norm(yy,uu)
    print("y norm:",gy)
    print("u norm:",gu)
    print("min-max normalized gain:",gy/gu)

    yy=(y-ys)
    uu=u
    gy,gu=compute_norm(yy,uu)
    print("y norm:",gy)
    print("u norm:",gu)
    print("gain:",gy/gu)

for i in range(8):
    print("=== {} ===".format(i))
    check(i)


