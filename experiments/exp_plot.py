import numpy as np
import joblib
import json
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
import argparse
from matplotlib import pylab as plt
from matplotlib import animation

def get_data(result_path):
    filename=result_path+"/sim/field_pt.npy"
    print("[LOAD]", filename)
    pt = np.load(filename)
    filename=result_path+"/sim/field_vec.npy"
    vec = np.load(filename)
    x=pt[:,0]
    y=pt[:,0]+vec[:,0]
    fx=(y-x)
    return x,fx

result_path2="bistable/010000result_fgh_loss"
x,fx=get_data(result_path2)
plt.plot(x,fx,label="fgh+",color="red")
result_path3="bistable/010000result_fgh"
x,fx=get_data(result_path3)
plt.plot(x,fx*0.1,label="fgh",color="orange")
result_path4="bistable/010000result_f"
x,fx=get_data(result_path4)
plt.plot(x,fx,label="f",color="green")
result_path1="bistable/010000result_vanilla"
x,fx=get_data(result_path1)
plt.plot(x,fx,label="vanilla",color="blue")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
filename="./exp_field_1d.eps"
print("[SAVE]",filename)
plt.savefig(filename)
plt.clf()
    
