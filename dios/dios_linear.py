import numpy as np
import joblib
import json
import logging
import dios
import dios.linear
import dios.AR
from dios.data_util import load_data
from dios.dios import get_default_config, build_config, set_file_logger
import argparse
from matplotlib import pylab as plt


def run_pred_mode(config, logger):
    # ... loading data
    logger.info("... loading data")
    all_data = load_data(mode="test", config=config, logger=None)
    # ... confirmation of input data and dimensions
    model_name=config["method"] 
    print("method:", model_name)
    print("data_size:", all_data.num)
    print("observation dimension:", all_data.obs_dim)
    print("input dimension:", all_data.input_dim)
    print("state dimension:", all_data.state_dim)
    obs_dim = all_data.obs_dim
    input_dim = all_data.input_dim
    state_dim = all_data.state_dim
    # ... defining data
    y_test =all_data.obs
    u_test =all_data.input
    x_test =all_data.state
    ##
    # ... loading
    model_path=config["save_model_path"]
    filename = model_path+"/"+model_name+".pkl"
    print("[LOAD]", filename)
    model = joblib.load(filename)

    ####
    mean_error = model.score(u_test,y_test)
    logger.info("mean error: {}".format(mean_error))
    # チェックに使うデータを決める
    # 初期値の推定(最初の2点のデータのみ)
    obs_gen=[]
    ss_type = ["ORT", "MOESP", "ORT_auto", "MOESP_auto"]
    ar_type = ["ARX", "PWARX", "ARX_auto", "PWARX_auto"]
    n=len(u_test)
    for i_N in range(n):
        if model_name in ss_type:
            x0 = model.predict_initial_state(u_test[i_N],y_test[i_N])
        elif model_name in ar_type:
            x0 = model.predict_initial_state(y_test[i_N])
        else:
            print("unknown model:", model_name)
        #  予測 
        y_hat =model.predict(x0,u_test[i_N])
        obs_gen.append(y_hat)
    obs_gen=np.array(obs_gen)
    ####
    print("=== gain")
    if all_data.stable is not None:
        print("Enabled stable observation")
        obs_stable=all_data.stable
        yy_data=np.sum((all_data.obs-obs_stable)**2,axis=2)
        yy_gen =np.sum((obs_gen     -obs_stable)**2,axis=2)
    else:
        yy_data=np.sum((all_data.obs)**2,axis=2)
        yy_gen =np.sum((obs_gen     )**2,axis=2)
    
    gu=np.sum(all_data.input**2,axis=2)

    gy_data=np.sqrt(np.mean(yy_data,axis=1))
    gy_gen =np.sqrt(np.mean(yy_gen ,axis=1))
    gu     =np.sqrt(np.mean(gu,axis=1))
    logger.info("mean(gu): {}".format(np.mean(gu)))
    logger.info("mean(gy): {}".format(np.mean(gy_data)))
    logger.info("mean(gy)/mean(gu): {}".format(np.mean(gy_data)))
    g_data=gy_data/gu
    g_gen=gy_gen/gu
    g_data[g_data == np.inf] = np.nan
    g_gen[g_gen == np.inf] = np.nan
    logger.info("data io gain: {}".format(np.nanmean(g_data)))
    logger.info("test io gain: {}".format(np.nanmean(g_gen)))
    ##
    ## ...plotting
    np.random.seed(1234)
    idx_all=np.arange(n)
    np.random.shuffle(idx_all)
    for i_N in idx_all[:10]:
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(obs_gen[i_N], label = 'y_hat')
        plt.plot(y_test[i_N], label = 'y', linestyle = "dashed")
        if model_name in ss_type:
            plt.title(model_name + ': index = {0}, n={1}, k={2}, mean error = {3:.2e}'.format(i_N, model.n, model.k,mean_error))
        elif model_name in ar_type:
            plt.title(model_name + ': index = {0}, mean error = {1:.2e}'.format(i_N, mean_error))
        elif model_name in ar_type:
            plt.title(model_name + ': index = {0}, mean error = {1:.2e}'.format(i_N, mean_error))
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(u_test[i_N],label = 'u')
        plt.legend()

        plot_path=config["plot_path"]
        filename=plot_path+"/"+model_name+'{:04d}.png'.format(i_N)
        print("[SAVE]", filename)
        plt.savefig(filename,dpi=100)
    ###
    print("=== long time (zero input)")
    np.random.seed(1234)
    N=10
    n_step=1000
    n_temp=20
    u=np.zeros((N,n_step,input_dim))
    y0=np.random.normal(0,1,(N,n_temp,obs_dim))
    fig = plt.figure()
    for i in range(N):
        if model_name in ss_type:
            x0 = model.predict_initial_state(u[i,:n_temp],y0[i,:n_temp])
        elif model_name in ar_type:
            x0 = model.predict_initial_state(y0[i])
        else:
            print("unknown model:", model_name)
        y =model.predict(x0,u[i])
        plt.plot(y)


    plot_path=config["plot_path"]
    filename=plot_path+"/"+model_name+'_zero.png'
    print("[SAVE]", filename)
    plt.savefig(filename,dpi=100)
    ###
    print("=== long time (random input)")
    np.random.seed(1234)
    u =np.random.normal(0,1,(N,n_step,input_dim))
    y0=np.zeros((N,n_temp,obs_dim))
    fig = plt.figure()
    plt.subplot(2,1,1)
    for i in range(N):
        if model_name in ss_type:
            x0 = model.predict_initial_state(u[i,:n_temp],y0[i,:n_temp])
        elif model_name in ar_type:
            x0 = model.predict_initial_state(y0[i])
        else:
            print("unknown model:", model_name)
        y =model.predict(x0,u[i])
        plt.plot(y)

    plt.subplot(2,1,2)
    for i in range(N):
        plt.plot(u[i])

    plot_path=config["plot_path"]
    filename=plot_path+"/"+model_name+'_rand.png'
    print("[SAVE]", filename)
    plt.savefig(filename,dpi=100)
    ###

def run_train_mode(config, logger):
    # ... loading data
    logger.info("... loading data")
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
    elif config["method"] == "ARX":
        #nty=2,ntu=2,N_max = 1000,N_alpha = 10,max_class = 25
        model=dios.AR.ARX()
        model.fit(u_train,y_train)
    elif config["method"] == "PWARX":
        model=dios.AR.PWARX()
        model.fit(u_train,y_train)
    elif config["method"] == "ORT_auto":
        model=dios.linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    elif config["method"] == "MOESP_auto":
        model=dios.linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    elif config["method"] == "ARX_auto":
        model=dios.AR.ARX()
        nty,ntu=model.autofit(u_train,y_train)
    elif config["method"] == "PWARX_auto":
        model=dios.AR.PWARX()
        nty,ntu=model.autofit(u_train,y_train)
    else:
        print("unknown method:",config["method"])
        exit()
    ##
    mean_error = model.score(u_train,y_train)
    logger.info("mean error: {}".format(mean_error))
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    config["method"]=args.method
    # setup
    mode_list = args.mode.split(",")
    for mode in mode_list:
        # mode
        if mode == "train":
            set_file_logger(logger,config,"log_linear_train."+config["method"]+".txt")
            run_train_mode(config, logger)
        elif mode == "infer" or mode == "test":
            set_file_logger(logger,config,"log_linear_test."+config["method"]+".txt")
            run_pred_mode(config, logger)

if __name__ == "__main__":
    main()
