import numpy as np
import joblib
import json
import dios
import dios.linear
from dios.data_util import load_data
from dios.dios import get_default_config, build_config
import argparse
from matplotlib import pylab as plt


def run_pred_mode(config):
    # ... loading data
    test_data = load_data(mode="test", config=config, logger=None)
    # ... confirmation of input data and dimensions
    model_name=config["method"] 
    print("method:", model_name)
    print("data_size:", test_data.num)
    print("observation dimension:", test_data.obs_dim)
    print("input dimension:", test_data.input_dim)
    print("state dimension:", test_data.state_dim)
    obs_dim = test_data.obs_dim
    # ... defining data
    y_test =test_data.obs
    u_test =test_data.input
    x_test =test_data.state
    ##
    # ... loading
    model_path=config["save_model_path"]
    filename = model_path+"/"+model_name+".pkl"
    print("[LOAD]", filename)
    model = joblib.load(filename)
    n=model.n
    k=model.k
    ###
    mean_error = model.score(u_test,y_test)
    print ("mean error:",mean_error)
    # チェックに使うデータを決める
    i_N =  0
    # 初期値の推定(最初の2点のデータのみ)
    x0 = model.predict_initial_state(u_test[i_N,:n],y_test[i_N,:n])
    #  予測 
    y_hat =model.predict(x0,u_test[i_N])
    ##
    # ...plotting
    plt.subplot(2,1,1)
    plt.plot(y_hat,label = 'y_hat')
    plt.plot(y_test[i_N],label = 'y', linestyle = "dashed")
    plt.title(model_name + ': n={0}, k={1}, mean error = {2:.2e}'.format(n,k,mean_error))
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(u_test[i_N],label = 'u')
    plt.legend()

    plot_path=config["plot_path"]
    filename=plot_path+"/"+model_name+'.png'
    print("[SAVE]", filename)
    plt.savefig(filename,dpi=100)


def run_train_mode(config):
    # ... loading data
    train_data = load_data(mode="train", config=config, logger=None)
    # ... confirmation of input data and dimensions
    model_name=config["method"] 
    print("method:", model_name)
    print("data_size:", train_data.num)
    print("observation dimension:", train_data.obs_dim)
    print("input dimension:", train_data.input_dim)
    print("state dimension:", train_data.state_dim)
    obs_dim = train_data.obs_dim
    # ... defining data
    y_train=train_data.obs
    u_train=train_data.input
    x_train=train_data.state
    # ... training
    n = config["state_dim"]
    k = 20
    if config["method"] == "ORT":
        model=dios.linear.ORT(n=n, k =k,initial_state =  'estimate')
        model.fit(u_train,y_train)
    elif config["method"] == "MOESP":
        model=dios.linear.MOESP(n=n, k =k,initial_state =  'estimate')
        model.fit(u_train,y_train)
    elif config["method"] == "ORT_auto":
        model=dios.linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    elif config["method"] == "MOESP_auto":
        model=dios.linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    else:
        print("unknown method:",config["method"])
        exit()
    ##
    mean_error = model.score(u_train,y_train)
    print ("mean error:",mean_error)
    # ...saving
    model_path=config["save_model_path"]
    filename = model_path+"/"+model_name+".pkl"
    print("[SAVE]", filename)
    joblib.dump(model, filename)
    ##
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/infer")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--method", type=str, default="MOESP", nargs="?", help="config json file"
    )
    parser.add_argument(
        "--save_config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument("--no-config", action="store_true", help="use default setting")
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument(
        "--hyperparam",
        type=str,
        default=None,
        nargs="?",
        help="hyperparameter json file",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument("--profile", action="store_true", help="")
    ## config
    for key, val in get_default_config().items():
        if type(val) is int:
            parser.add_argument("--"+key, type=int, default=val, help="[config integer]")
        elif type(val) is float:
            parser.add_argument("--"+key, type=float, default=val, help="[config float]")
        elif type(val) is bool:
            parser.add_argument("--"+key, action="store_true", help="[config bool]")
        else:
            parser.add_argument("--"+key, type=str, default=val, help="[config string]")
    args = parser.parse_args()
    # config
    config = get_default_config()
    for key, val in get_default_config().items():
        config[key]=getattr(args,key)
    # 
    if args.config is None:
        if not args.no_config:
            parser.print_help()
            quit()
    else:
        print("[LOAD]",args.config)
        fp = open(args.config, "r")
        config.update(json.load(fp))
    build_config(config)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    if "log" in config:
        h = logging.FileHandler(filename=config["log"], mode="w")
        h.setLevel(logging.INFO)
        logger.addHandler(h)
    """
    config["method"]=args.method
    # setup
    mode_list = args.mode.split(",")
    for mode in mode_list:
        # mode
        if mode == "train":
            run_train_mode(config)
        elif mode == "infer" or mode == "test":
            run_pred_mode(config)

if __name__ == "__main__":
    main()
