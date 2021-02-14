import numpy as np
import glob
import os
from numba import jit
from dios.dios import NumPyArangeEncoder
import json

def minmax_normalize(x_data,u_data,y_data,ys_data, path="dataset", name="none"):
    x_max=np.max(x_data[:,:,:])
    y_max=np.max(y_data[:,:,:])
    u_max=np.max(u_data[:,:,:])
    x_min=np.min(x_data[:,:,:])
    y_min=np.min(y_data[:,:,:])
    u_min=np.min(u_data[:,:,:])
    x_data=(x_data-x_min)/(x_max-x_min)
    y_data=(y_data-y_min)/(y_max-y_min)
    u_data=(u_data-u_min)/(u_max-u_min)
    if ys_data is not None:
        ys_data=(ys_data-y_min)/(y_max-y_min)
    minmax_data={"name":name,
            "x_max":x_max,"y_max":y_max,"u_max":u_max,
            "x_min":x_min,"y_min":y_min,"u_min":u_min,
            }
    filename=path+"/minmax_data.json"
    os.makedirs(path,exist_ok=True)
    json.dump(minmax_data,open(filename,"w"),
            cls=NumPyArangeEncoder)
    print("[SAVE]",filename)
    return x_data, u_data, y_data, ys_data


def save_dataset(x_data, u_data, y_data, ys_data, M, path="dataset", name="none"):
    os.makedirs(path,exist_ok=True)
    filename=path+"/"+name+".train.obs.npy"
    print("[SAVE]",filename)
    print(y_data[:M].shape)
    np.save(filename,y_data[:M])
    filename=path+"/"+name+".test.obs.npy"
    print("[SAVE]",filename)
    print(y_data[M:].shape)
    np.save(filename,y_data[M:])

    filename=path+"/"+name+".train.input.npy"
    print("[SAVE]",filename)
    print(u_data[:M].shape)
    np.save(filename,u_data[:M])
    filename=path+"/"+name+".test.input.npy"
    print("[SAVE]",filename)
    print(u_data[M:].shape)
    np.save(filename,u_data[M:])

    filename=path+"/"+name+".train.state.npy"
    print("[SAVE]",filename)
    print(x_data[:M].shape)
    np.save(filename,x_data[:M])
    filename=path+"/"+name+".test.state.npy"
    print("[SAVE]",filename)
    print(x_data[M:].shape)
    np.save(filename,x_data[M:])

    if ys_data is not None:
        filename=path+"/"+name+".train.stable.npy"
        print("[SAVE]",filename)
        print(ys_data[:M].shape)
        np.save(filename,ys_data[:M])
        filename=path+"/"+name+".test.stable.npy"
        print("[SAVE]",filename)
        print(ys_data[M:].shape)
        np.save(filename,ys_data[M:])


