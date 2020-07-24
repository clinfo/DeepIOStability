import numpy as np
import glob
import os
#GlucoseInsulin00_u_data.npy
N=100
L=300
for filename in glob.glob("data/*_u_data.npy"):
    y=np.load(filename)
    #name=os.path.basename(filename)
    name=filename.split("_")[0]
    y=y[:N,:L,:]
    print(y.shape)
    M=int(y.shape[0]*0.8)
    print(name)
    my=np.max(y)
    name1=name+".train.input.npy"
    name2=name+".test.input.npy"
    np.save(name1,y[:M,:,:]/my)
    np.save(name2,y[M:,:,:]/my)
    #np.load(x)

for filename in glob.glob("data/*_y_data.npy"):
    y=np.load(filename)
    #name=os.path.basename(filename)
    name=filename.split("_")[0]
    y=y[:N,:L,:]
    print(y.shape)
    M=int(y.shape[0]*0.8)
    print(name)
    my=np.max(y)
    print("mean: ",np.mean(y))
    print("max : ",np.max(y))
    name1=name+".train.obs.npy"
    name2=name+".test.obs.npy"
    np.save(name1,y[:M,:,:]/my)
    np.save(name2,y[M:,:,:]/my)
    #np.load(x)

for filename in glob.glob("data/*_x_data.npy"):
    y=np.load(filename)
    #name=os.path.basename(filename)
    name=filename.split("_")[0]
    y=y[:N,:L,:]
    print(y.shape)
    M=int(y.shape[0]*0.8)
    print(name)
    my=np.max(y)
    name1=name+".train.state.npy"
    name2=name+".test.state.npy"
    np.save(name1,y[:M,:,:]/my)
    np.save(name2,y[M:,:,:]/my)
    #np.load(x)

