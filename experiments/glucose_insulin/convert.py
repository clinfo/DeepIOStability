import numpy as np
import glob
import os
import json
#GlucoseInsulin00_u_data.npy
N=100
L=300
r=0.8
def save(name,data,out_type):
    path="data/"
    data=data[:N,:L,:]
    M=int(data.shape[0]*r)
    name1=path+name+".train."+out_type+".npy"
    name2=path+name+".test."+out_type+".npy"
    np.save(name1,data[:M,:,:])
    np.save(name2,data[M:,:,:])
    print("[SAVE]",name1)
    print("[SAVE]",name2)


def convert(param_index):
    name="GlucoseInsulin{:02d}".format(param_index)
    filename="data_org/"+name+"_u_data.npy"
    u=np.load(filename)
    filename="data_org/"+name+"_x_data.npy"
    x=np.load(filename)
    filename="data_org/"+name+"_y_data.npy"
    y=np.load(filename)
    filename="data_org/"+name+"_ys_data.npy"
    ys=np.load(filename)

    #x_max=np.max(x[:,:,:],axis=(0,1))
    x_max=np.max(x[:,:,:])
    # stable x0 == 0 
    print("max x:",x_max)
    save(name,x/x_max,"state")
    
    #y_max=np.max(y[:,:,:],axis=(0,1))
    y_max=np.max(y[:,:,:])
    #y=y-ys
    print("max y:",y_max)
    save(name,y/y_max,"obs")
    save(name,ys/y_max,"stable")
 
    #u_max=np.max(u[:,:,:],axis=(0,1))
    u_max=np.max(u[:,:,:])
    print("max u:",u_max)
    save(name,u/u_max,"input")
    return {"name":name,"x_max":x_max,"y_max":y_max,"u_max":u_max}
    #return {"name":name,"x_max":list(x_max),"y_max":list(y_max),"u_max":list(u_max)}


max_data={}
for i in range(9):
    print("=== {} ===".format(i))
    m=convert(i)
    max_data[i]=m
json.dump(max_data,open("data/max_data.json","w"))
