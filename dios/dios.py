#
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import logging

from six.moves import urllib
import numpy as np
import joblib
import json
import argparse

from dios.data_util import load_data
from dios.model import DiosSSM
from dios.simple_system import SimpleSystem
import torch
import torch.nn as nn
import torch.nn.functional as F


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


def build_config(config):
    # backward compatibility
    for key in config.keys():
        if "_npy" in config.keys():
            new_key = key.replace("_npy", "")
            config[new_key] = config[key]

    if "dim" in config:
        config["state_dim"] = config["dim"]
    if "result_path" in config:
        path = config["result_path"]
        os.makedirs(path, exist_ok=True)
        os.makedirs(path+"/model", exist_ok=True)
        os.makedirs(path+"/plot", exist_ok=True)
        os.makedirs(path+"/sim", exist_ok=True)
        config["save_model_path"] = path + "/model"
        config["save_result_test"] = path + "/test.jbl"
        config["save_result_train"] = path + "/train.jbl"
        config["simulation_path"] = path + "/sim"
        config["load_model"] = path + "/model/best.checkpoint"
        config["plot_path"] = path + "/plot"
        config["log_path"] = path


def get_default_config():
    config = {}
    # data and network
    # config["dim"]=None
    config["state_dim"] = 2
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100
    #config["activation"] = "relu"
    #config["optimizer"] = "sgd"
    ##
    config["v_type"] = "single"
    config["system_scale"] = 0.1
    config["learning_rate"] = 1.0e-2
    # dataset
    config["train_valid_ratio"] = 0.2
    config["data_train"] = None
    config["data_test"] = None
    # save/load model
    config["save_model_path"] = None
    config["load_model"] = None
    config["save_result_train"] = None
    config["save_result_test"] = None
    config["save_result_filter"] = None

    config["delta_t"]=0.1
    config["gamma"]=None
    config["c"]=0.1
    config["init_state_mode"]="estimate_state"
    config["alpha_recons"]=1.0
    config["alpha_HJ"]=1.0
    config["alpha_HJ_dvf"]=1.0
    config["alpha_HJ_hh"]=1.0
    config["alpha_HJ_gg"]=1.0
    config["alpha_gamma"]=1.0
    config["alpha_state"]=1.0
    config["hj_loss_type"]="const"
    config["diag_g"]=True
    
    config["weight_decay"] = 0.01
    config["hidden_layer_f"] = [32]
    config["hidden_layer_g"] = [32]
    config["hidden_layer_h"] = [32]
    """
    config["alpha"] = 1.0
    config["beta"] = 1.0
    config["gamma"] = 1.0
    ##
    config["curriculum_alpha"] = False
    config["curriculum_beta"] = False
    config["curriculum_gamma"] = False
    config["sampling_tau"] = 10  # 0.1
    config["normal_max_var"] = 5.0  # 1.0
    config["normal_min_var"] = 1.0e-5
    config["zero_dynamics_var"] = 1.0
    config["pfilter_sample_size"] = 10
    config["pfilter_proposal_sample_size"] = 1000
    config["pfilter_save_sample_num"] = 100
    config["label"] = "multinominal"
    config["task"] = "generative"
    # config["state_type"]="discrete"
    config["state_type"] = "normal"
    config["sampling_type"] = "none"
    config["time_major"] = True
    config["steps_train_npy"] = None
    config["steps_test_npy"] = None
    config["sampling_type"] = "normal"
    config["emission_type"] = "normal"
    config["state_type"] = "normal"
    config["dynamics_type"] = "distribution"
    config["pfilter_type"] = "trained_dynamics"
    config["potential_enabled"] = (True,)
    config["potential_grad_transition_enabled"] = (True,)
    config["potential_nn_enabled"] = (False,)
    config["potential_grad_delta"] = 0.1
    #
    config["field_grid_num"] = 30
    config["field_grid_dim"] = None
    """
    # generate json
    return config


def run_train_mode(config, logger):
    logger.info("... loading data")
    all_data = load_data(mode="train", config=config, logger=logger)
    train_data, valid_data = all_data.split(1.0 - config["train_valid_ratio"])
    print("train_data_size:", train_data.num)

    # defining dimensions from given data
    print("observation dimension:", train_data.obs_dim)
    print("input dimension:", train_data.input_dim)
    print("state dimension:", train_data.state_dim)
    input_dim = train_data.input_dim if train_data.input_dim is not None else 1
    state_dim = config["state_dim"]
    obs_dim = train_data.obs_dim
    #
    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
    else:
        device = 'cpu'
        print("device: cpu")
    # defining system
    hidden_layer_h=config["hidden_layer_h"]
    hidden_layer_f=config["hidden_layer_f"]
    hidden_layer_g=config["hidden_layer_g"]
    sys = SimpleSystem(obs_dim, state_dim, input_dim,
            hidden_layer_h=hidden_layer_h,
            hidden_layer_f=hidden_layer_f,
            hidden_layer_g=hidden_layer_g,
            delta_t=config["delta_t"],
            gamma=config["gamma"],
            c=config["c"],
            init_state_mode=config["init_state_mode"],
            alpha={
                "recons":config["alpha_recons"],
                "HJ":config["alpha_HJ"],
                "gamma":config["alpha_gamma"],
                "state":config["alpha_state"],
                "HJ_dvf":config["alpha_HJ_dvf"],
                "HJ_hh":config["alpha_HJ_hh"],
                "HJ_gg":config["alpha_HJ_gg"],},
            hj_loss_type=config["hj_loss_type"],
            diag_g=config["diag_g"],
            scale=config["system_scale"],
            v_type=config["v_type"],
            device=device
            )
    # training NN from data
    model = DiosSSM(config, sys, device=device)
    train_loss,valid_loss=model.fit(train_data, valid_data)
    joblib.dump(train_loss, config["save_model_path"]+"/train_loss.pkl")
    joblib.dump(valid_loss, config["save_model_path"]+"/valid_loss.pkl")

    model.load_ckpt(config["save_model_path"]+"/best.checkpoint")
    loss, states, obs_gen = model.simulate_with_data(valid_data)
    save_simulation(config,valid_data,states,obs_gen)
    joblib.dump(loss, config["simulation_path"]+"/last_loss.pkl")
    
    
def save_simulation(config,data,states,obs_gen):
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/obs.npy"
        print("[SAVE]", filename)
        np.save(filename, data.obs)
        if data.state is not None:
            filename=config["simulation_path"]+"/state_true.npy"
            print("[SAVE]", filename)
            np.save(filename, data.state)
        if data.input is not None:
            filename=config["simulation_path"]+"/input.npy"
            print("[SAVE]", filename)
            np.save(filename, data.input)
        filename=config["simulation_path"]+"/obs_gen.npy"
        print("[SAVE]", filename)
        np.save(filename, obs_gen.to("cpu").detach().numpy().copy())
        filename=config["simulation_path"]+"/states.npy"
        print("[SAVE]", filename)
        np.save(filename, states.to("cpu").detach().numpy().copy())


def run_pred_mode(config, logger):
    logger.info("... loading data")
    all_data = load_data(mode="test", config=config, logger=logger)
    print("data_size:", all_data.num)

    # defining dimensions from given data
    print("observation dimension:", all_data.obs_dim)
    print("input dimension:", all_data.input_dim)
    print("state dimension:", all_data.state_dim)
    input_dim = all_data.input_dim if all_data.input_dim is not None else 1
    state_dim = config["state_dim"]
    obs_dim = all_data.obs_dim
    #
    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
    else:
        device = 'cpu'
        print("device: cpu")
    # defining system
    hidden_layer_h=config["hidden_layer_h"]
    hidden_layer_f=config["hidden_layer_f"]
    hidden_layer_g=config["hidden_layer_g"]
    sys = SimpleSystem(obs_dim, state_dim, input_dim,
            hidden_layer_h=hidden_layer_h,
            hidden_layer_f=hidden_layer_f,
            hidden_layer_g=hidden_layer_g,
            delta_t=config["delta_t"],
            gamma=config["gamma"],
            c=config["c"],
            init_state_mode=config["init_state_mode"],
            alpha={
                "recons":config["alpha_recons"],
                "HJ":config["alpha_HJ"],
                "gamma":config["alpha_gamma"],
                "state":config["alpha_state"],
                "HJ_dvf":config["alpha_HJ_dvf"],
                "HJ_hh":config["alpha_HJ_hh"],
                "HJ_gg":config["alpha_HJ_gg"],},
            hj_loss_type=config["hj_loss_type"],
            diag_g=config["diag_g"],
            scale=config["system_scale"],
            v_type=config["v_type"],
            device=device
            )

    # training NN from data
    model = DiosSSM(config, sys, device=device)
    model.load_ckpt(config["load_model"])
    logger.info("... simulating data")
    loss, states, obs_gen = model.simulate_with_data(all_data, step_wise_loss=True)
    save_simulation(config,all_data,states,obs_gen)
    obs_gen=obs_gen.to("cpu").detach().numpy().copy()
    ##
    x=np.sum((all_data.obs-obs_gen)**2,axis=2)
    x=np.mean(x,axis=1)
    mse=np.mean(x,axis=0)
    logger.info("... loading data")
    logger.info("mean error: {}".format(mse))
    ####
    print("=== gain")
    if all_data.input is not None:
        if all_data.stable is not None:
            print("Enabled stable observation")
            obs_stable=all_data.stable
            yy_data=np.sum((all_data.obs-obs_stable)**2,axis=2)
            yy_gen =np.sum((obs_gen     -obs_stable)**2,axis=2)
        else:
            yy_data=np.sum((all_data.obs)**2,axis=2)
            yy_gen =np.sum((obs_gen     )**2,axis=2)
        
        gu=np.sum(all_data.input**2,axis=2)

        gy_data=np.mean(np.mean(yy_data,axis=1),axis=0)
        gy_gen =np.mean(np.mean(yy_gen ,axis=1),axis=0)
        gu=np.mean(np.mean(gu,axis=1),axis=0)
        logger.info("data io gain: {}".format(gy_data/gu))
        logger.info("test io gain: {}".format(gy_gen/gu))
    ####
    print("=== stable")
    st_pts,st_mu=model.system_model.get_stable()
    st_errors=[]
    for pt,mu in zip(st_pts,st_mu):
        mu=mu.to("cpu").detach().numpy().copy()
        pt=pt.to("cpu").detach().numpy().copy()
        logger.info("mu: {}".format(str(mu)))
        logger.info("stable point h(mu): {}".format(str(pt)))
        if all_data.stable is not None:
            obs_stable=all_data.stable
            logger.info("data stable: {}".format(str(obs_stable[0,0,:])))
            e=(obs_stable[:,:,:]-pt)
            st_errors.append(e**2)
    if len(st_errors)>0:
        st_e=np.stack(st_errors, axis=0)
        st_e=np.min(st_e,axis=0)
        st_e=np.mean(st_e)
        logger.info("stable error: {}".format(str(st_e)))
    ####
    print("=== field")
    pt,vec=model.get_vector_field(state_dim, dim=[0,1],min_v=-3,max_v=3,delta=0.2)
    print(x)
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/field_pt.npy"
        print("[SAVE]", filename)
        np.save(filename, pt)
        filename=config["simulation_path"]+"/field_vec.npy"
        print("[SAVE]", filename)
        np.save(filename, vec)
    ####
    N=10
    n_step=1000
    print("=== long time (zero input)")
    init_state=torch.tensor(np.random.normal(0,1,(N,state_dim)),dtype=torch.float32)
    out_state,out_obs=model.simulate_with_input(None, init_state, step=n_step)
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/sim_zero.obs.npy"
        print("[SAVE]", filename)
        np.save(filename, out_obs)
        filename=config["simulation_path"]+"/sim_zero.state.npy"
        print("[SAVE]", filename)
        np.save(filename, out_state)
    ####
    print("=== long time (random input)")
    input_data=torch.tensor(np.random.normal(0,1,(N,n_step,input_dim)),dtype=torch.float32)
    init_state=torch.tensor(np.random.normal(0,1,(N,state_dim)),dtype=torch.float32)
    out_state,out_obs=model.simulate_with_input(input_data, init_state, step=n_step)
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/sim_rand.obs.npy"
        print("[SAVE]", filename)
        np.save(filename, out_obs)
        filename=config["simulation_path"]+"/sim_rand.state.npy"
        print("[SAVE]", filename)
        np.save(filename, out_state)
     
def set_file_logger(logger,config,filename):
    if "log_path" in config:
        filename=config["log_path"]+"/"+filename
        h = logging.FileHandler(filename=filename, mode="w")
        h.setLevel(logging.INFO)
        logger.addHandler(h)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/infer")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
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
            parser.add_argument("--"+key, type=bool, default=val, help="[config float]")
            #parser.add_argument("--"+key, action="store_true", help="[config bool]")
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
    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # profile
    config["profile"] = args.profile
    #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")

    # setup
    mode_list = args.mode.split(",")
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    for mode in mode_list:
        # mode
        if mode == "train":
            set_file_logger(logger,config,"log_train.txt")
            run_train_mode(config, logger)
        elif mode == "infer" or mode == "test":
            set_file_logger(logger,config,"log_test.txt")
            if args.model is not None:
                config["load_model"] = args.model
            run_pred_mode(config, logger)
        elif mode == "filter":
            if args.model is not None:
                config["load_model"] = args.model
            filtering(sess, config)
        elif mode == "filter_discrete":
            filter_discrete_forward(sess, config)
        elif mode == "train_fivo":
            train_fivo(sess, config)
        elif mode == "field":
            field(sess, config)
        elif mode == "potential":
            potential(sess, config)
        elif args.mode == "filter_server":
            filtering_server(sess, config=config)

    if args.save_config is not None:
        print("[SAVE] config: ", args.save_config)
        fp = open(args.save_config, "w")
        json.dump(
            config,
            fp,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            cls=NumPyArangeEncoder,
        )


if __name__ == "__main__":
    main()
