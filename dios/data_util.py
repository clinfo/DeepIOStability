import os
import numpy as np

import torch


class DiosDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, train=True):
        super(DiosDataset, self).__init__()
        self.transform = transform
        self.data = data

    def __len__(self):
        return self.data.num

    def __getitem__(self, idx):
        # out_obs, out_input, out_state=None,None,None
        out_obs, out_input, out_state = 0, 0, 0
        out_obs = self.data.obs[idx]
        if self.data.input is not None:
            out_input = self.data.input[idx]
        if self.data.state is not None:
            out_state = self.data.state[idx]

        if self.transform:
            out_obs = self.transform(out_obs)

        return out_obs, out_input, out_state


class DiosData:
    def __init__(self):
        self.obs = None
        self.obs_mask = None
        self.obs_dim = None
        self.input = None
        self.input_mask = None
        self.input_dim = None
        self.state = None
        self.state_mask = None
        self.state_dim = None
        self.step = None
        self.idx = None

    def split(self, rate):
        idx = list(range(self.num))
        np.random.shuffle(idx)
        m = int(self.num * rate)
        idx1 = idx[:m]
        idx2 = idx[m:]
        data1 = DiosData()
        data2 = DiosData()
        copy_attrs = ["obs_dim", "input_dim", "state_dim"]
        split_attrs = [
            "obs",
            "obs_mask",
            "input",
            "input_mask",
            "state",
            "state_mask",
            "idx",
        ]
        ## split
        for attr in split_attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(data1, attr, val[idx1])
                setattr(data2, attr, val[idx2])
        data1.num = m
        data2.num = self.num - m
        ## copy
        for attr in copy_attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(data1, attr, val)
                setattr(data2, attr, val)
        return data1, data2

    def set_dim_from_data(self):
        attrs = ["obs", "input", "state"]
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr + "_dim", val.shape[2])


def np_load(filename, logger):
    logger.info("[LOAD] " + filename)
    return np.load(filename)


def load_data(mode, config, logger):
    name = config["data_" + mode]
    if os.path.exists(name):
        return load_simple_data(name, config, logger)
    else:
        return load_all_data(name, config, logger)


def load_all_data(name, config, logger):
    data = DiosData()
    data.obs = np_load(name + ".obs.npy", logger)
    data.num = data.obs.shape[0]
    data.idx = np.array(list(range(data.num)))
    ###
    filename = name + ".obs_mask.npy"
    if os.path.exists(filename):
        data.obs_mask = np_load(filename, logger)
    else:
        data.obs_mask = np.ones_like(data.obs)
    ###
    filename = name + ".step.npy"
    if os.path.exists(filename):
        data.step = np_load(filename, logger)
    else:
        s = data.obs.shape[1]
        data.step = np.array([s] * data.num)
    ###
    keys = ["input", "state"]
    for key in keys:
        filename = name + "." + key + ".npy"
        val = None
        if os.path.exists(filename):
            val = np_load(filename, logger)
            setattr(data, key, val)
            valid_flag = True
        else:
            setattr(data, key, None)
        ###
        filename = name + "." + key + "_mask.npy"
        if os.path.exists(filename):
            setattr(data, key + "_mask", np_load(filename, logger))
        if val is not None:
            setattr(data, key + "_mask", np.ones_like(val))
        else:
            setattr(data, key + "_mask", None)
        ###
    data.set_dim_from_data()
    return data


def load_simple_data(filename, config, logger):
    data = DiosData()
    data.obs = np_load(filename, logger)
    data.obs_mask = np.ones_like(data.obs)
    data.num = data.obs.shape[0]
    data.idx = np.array(list(range(data.num)))
    s = data.obs.shape[1]
    data.step = np.array([s] * data.num)
    data.set_dim_from_data()
    return data
