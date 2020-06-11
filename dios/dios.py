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
        config["load_model"] = path + "/model/model.best.ckpt"
        config["plot_path"] = path + "/plot"
        config["log"] = path + "/log.txt"


def get_default_config():
    config = {}
    # data and network
    # config["dim"]=None
    config["state_dim"] = 2
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100
    ##
    config["alpha"] = 1.0
    config["beta"] = 1.0
    config["gamma"] = 1.0
    config["learning_rate"] = 1.0e-2
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
    # dataset
    config["train_valid_ratio"] = 0.2
    config["data_train"] = None
    config["data_test"] = None
    config["label"] = "multinominal"
    config["task"] = "generative"
    # save/load model
    config["save_model_path"] = None
    config["load_model"] = None
    config["save_result_train"] = None
    config["save_result_test"] = None
    config["save_result_filter"] = None
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
    input_dim = train_data.input_dim
    state_dim = config["state_dim"]
    obs_dim = train_data.obs_dim
    # defining system
    hidden_layer_dim = 32
    sys = SimpleSystem(obs_dim, state_dim, input_dim, hidden_layer_dim=hidden_layer_dim)

    # training NN from data
    model = DiosSSM(config, sys)
    model.fit(train_data, valid_data)
    model.load_ckpt(config["save_model_path"]+"/best.checkpoint")

    loss, states, obs_gen = model.simulate_with_data(valid_data)
    save_simulation(config,valid_data,states,obs_gen)
    
def save_simulation(config,data,states,obs_gen):
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/obs.npy"
        print("[LOAD]", filename)
        np.save(filename, data.obs)
        filename=config["simulation_path"]+"/obs_gen.npy"
        print("[LOAD]", filename)
        np.save(filename, obs_gen.to("cpu").detach().numpy().copy())
        filename=config["simulation_path"]+"/states.npy"
        print("[LOAD]", filename)
        np.save(filename, states.to("cpu").detach().numpy().copy())


def run_pred_mode(config, logger):
    logger.info("... loading data")
    all_data = load_data(mode="test", config=config, logger=logger)
    print("data_size:", all_data.num)

    # defining dimensions from given data
    print("observation dimension:", all_data.obs_dim)
    print("input dimension:", all_data.input_dim)
    print("state dimension:", all_data.state_dim)
    input_dim = all_data.input_dim
    state_dim = config["state_dim"]
    obs_dim = all_data.obs_dim
    # defining system
    hidden_layer_dim = 32
    sys = SimpleSystem(obs_dim, state_dim, input_dim, hidden_layer_dim=hidden_layer_dim)

    # training NN from data
    model = DiosSSM(config, sys)
    model.load_ckpt(config["load_model"])

    loss, states, obs_gen = model.simulate_with_data(all_data)
    save_simulation(config,all_data,states,obs_gen)


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
    args = parser.parse_args()
    # config
    config = get_default_config()
    if args.config is None:
        if not args.no_config:
            parser.print_help()
            quit()
    else:
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
    if "log" in config:
        h = logging.FileHandler(filename=config["log"], mode="w")
        h.setLevel(logging.INFO)
        logger.addHandler(h)

    # setup
    mode_list = args.mode.split(",")
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    for mode in mode_list:
        # mode
        if mode == "train":
            run_train_mode(config, logger)
        elif mode == "infer" or mode == "test":
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
