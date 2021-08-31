import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_train,loss_valid,filename="loss.png"):
    ### loss.png
    train_loss_list=[]
    valid_loss_list=[]
    for train_l,valid_l in zip(loss_train.loss_history, loss_valid.loss_history):
        train_loss_list.append(train_l.item())
        valid_loss_list.append(valid_l.item())
    med_y=np.nanmedian(train_loss_list)
    min_y=np.nanmin(train_loss_list)


    plt.figure(figsize=(12,12))
    plt.plot(train_loss_list,label="train loss")
    plt.plot(valid_loss_list,label="validation loss")
    plt.legend()
    plt.ylim(min_y,med_y*3)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(filename)
    print("[SAVE]",filename)

def plot_loss_detail(loss_train,loss_valid,filename="loss_detail.png"):
    ### loss_detail.png
    train_loss_dict={}
    valid_loss_dict={}
    loss_list=[]
    for i,train_l in enumerate(loss_train.loss_dict_history):
        for key,val in train_l.items():
            if key not in train_loss_dict:
                train_loss_dict[key]=[]
            v=val.item()
            train_loss_dict[key].append((i,v))
            if type(v) is float: 
                loss_list.append(v)
            else:
                loss_list.extend(v)

    for i,valid_l in enumerate(loss_valid.loss_dict_history):
        for key,val in valid_l.items():
            if key not in valid_loss_dict:
                valid_loss_dict[key]=[]
            v=val.item()
            valid_loss_dict[key].append((i,v))
            if type(v) is float: 
                loss_list.append(v)
            else:
                loss_list.extend(v)

    med_y=np.nanmedian(train_loss_list)
    min_y=np.nanmin(loss_list)


    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(12,12))
    for i, (name, loss) in enumerate(train_loss_dict.items()):
        xs=[x for x,y in loss]
        ys=[y for x,y in loss]
        plt.plot(xs,ys,label="train:"+name,color=cmap(i%cmap.N))

    for i, (name, loss) in enumerate(valid_loss_dict.items()):
        xs=[x for x,y in loss]
        ys=[y for x,y in loss]
        plt.plot(xs,ys,label="valid:"+name,color=cmap(i%cmap.N), linestyle='dashed')


    plt.legend()
    plt.ylim(min_y,med_y*4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(filename)
    print("[SAVE]",filename)

def main():
    loss_train=pickle.load(open("model/train_loss.pkl","rb"))
    loss_valid=pickle.load(open("model/valid_loss.pkl","rb"))
    plot_loss(loss_train,loss_valid)
    plot_loss_detail(loss_train,loss_valid)


if __name__ == "__main__":
    main() 
