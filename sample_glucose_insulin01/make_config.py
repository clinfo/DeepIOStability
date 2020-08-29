import json
obj=json.load(open("config.json"))
for i in range(9):
    obj["data_train"]="data/GlucoseInsulin{:02d}.train".format(i)
    obj["data_test"] ="data/GlucoseInsulin{:02d}.test".format(i)
    obj["result_path"]="result{:02d}/".format(i)
    fp=open("config/config{:02d}.json".format(i),"w")
    json.dump(obj,fp)
    #print(obj)

#"learning_rate":0.01,
#"alpha_HJ":0.01,
#"alpha_gamma":0.1,
#"delta_t":1,
#"state_dim":6,
#"batch_size":100

