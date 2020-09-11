import json
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="config.json", nargs="?", help="config json file"
)
args = parser.parse_args()
obj=json.load(open(args.config))
name,ext=os.path.splitext(os.path.basename(args.config))
dir_name=os.path.dirname(obj["result_path"])
print(dir_name)
for i in range(9):
    obj["data_train"]="data/GlucoseInsulin{:02d}.train".format(i)
    obj["data_test"] ="data/GlucoseInsulin{:02d}.test".format(i)
    obj["result_path"]=dir_name+"{:02d}/".format(i)
    filename="config/"+name+"{:02d}.json".format(i)
    fp=open(filename,"w")
    print(filename)
    json.dump(obj,fp)
    #print(obj)

#"learning_rate":0.01,
#"alpha_HJ":0.01,
#"alpha_gamma":0.1,
#"delta_t":1,
#"state_dim":6,
#"batch_size":100

