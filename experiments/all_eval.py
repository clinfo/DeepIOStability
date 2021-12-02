import glob
import os
import re

filenames=glob.glob("./**/eval.tsv")
header=None
data=[]
for filename in filenames:
    name=os.path.basename(os.path.dirname(filename))
    fp=open(filename)
    head=next(fp)
    arr=head.strip().split("\t")
    if header is None:
        header=arr
    for line in fp:
        arr=line.split("\t")
        arr=list(map(lambda x: x.strip(),arr))
        method_key,_=os.path.splitext(os.path.basename(arr[0]))
        setting_key=os.path.dirname(arr[0])
        key1,key2=os.path.splitext(method_key)
        #print(key1)
        if key1=="log_linear_test":
            method1="linear"
            method2=key2[1:]
        elif key1=="log_test":
            method1="nn"
            method2=setting_key
            a=setting_key.split("_")
            if len(a)>1:
                method2="_".join(a[1:])
            
            #if "_base" in setting_key:
            #    method2="vanilla"
            #    arr[5]="" # for gamma 
            #else:
            #    method2="IO-Stability"
        setting_key=setting_key.split("_")[0]
        m=re.search(r'([0-9]+?)result',setting_key)
        if m:
            setting_name=m.group(1)
            setting_name=str(int(setting_name))
        else:
            setting_name=setting_key
        print(setting_name)
        data.append([name,setting_name,method1,method2]+arr)
ofp = open("all_eval.tsv","w")
s="\t".join(["name","setting","method_type","method"]+header)
ofp.write(s)
print(s)
ofp.write("\n")
for arr in sorted(data):
    s="\t".join(arr)
    print(s)
    ofp.write(s)
    ofp.write("\n")
