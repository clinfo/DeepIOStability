import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dios.data_util import DiosDataset


class LossLogger:
    def __init__(self):
        self.loss_history=[]
        self.loss_dict_history=[]

    def start_epoch(self):
        self.running_loss = 0
        self.running_loss_dict = {}
        self.running_count = 0

    def update(self, loss, loss_dict):
        self.running_loss += loss
        self.running_count +=1
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[k] += v
            else:
                self.running_loss_dict[k] = v

    def end_epoch(self):
        self.running_loss /= self.running_count
        for k in self.running_loss_dict.keys():
            self.running_loss_dict[k] /=  self.running_count
        self.loss_history.append(self.running_loss)
        self.loss_dict_history.append(self.running_loss_dict)

    def get_msg(self, prefix="train"):
        msg = []
        m = "{:s}-loss: {:.3f}".format(prefix, self.running_loss)
        msg.append(m)
        for k, v in self.running_loss_dict.items():
            m = "{:s}-{:s}-loss: {:.3f}".format(prefix, k, v)
            msg.append(m)
        return "  ".join(msg)


class DiosSSM:
    def __init__(self, config, system_model):
        self.config = config
        self.system_model = system_model

    def _compute_batch_simulate(self, batch):
        obs, input_, state = batch
        metrics = {}
        if input_ is 0:  ## to avoid error (specification of pytorch)
            input_ = None
        state_generated, obs_generated = self.system_model.forward_simulate(obs, input_, state)
        loss_dict = self.system_model.forward_loss(
            obs, input_, state_generated, obs_generated, state
        )
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss, loss_dict, state_generated, obs_generated

    def simulate_with_data(self, valid_data):
        config = self.config
        validset = DiosDataset(valid_data, train=False)
        batch_size = config["batch_size"]
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=2, timeout=10
        )

        valid_loss_logger = LossLogger()
        valid_loss_logger.start_epoch()
        state_generated_list, obs_generated_list = [], []
        for i, batch in enumerate(validloader, 0):
            loss, loss_dict, state_generated, obs_generated = self._compute_batch_simulate(batch)
            state_generated_list.append(state_generated)
            obs_generated_list.append(obs_generated)
            valid_loss_logger.update(loss, loss_dict)
        valid_loss_logger.end_epoch()

        print(valid_loss_logger.get_msg("valid"))
        out_state_generated = torch.cat(state_generated_list, dim=0)
        out_obs_generated = torch.cat(obs_generated_list, dim=0)
        return valid_loss_logger, out_state_generated, out_obs_generated

    def _compute_batch_loss(self, batch):
        obs, input_, state = batch
        metrics = {}
        if input_ is 0:  ## to avoid error (specification of pytorch)
            input_ = None
        loss_dict = self.system_model(obs, input_, state)
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss, loss_dict

    def fit(self, train_data, valid_data):
        config = self.config
        batch_size = config["batch_size"]
        trainset = DiosDataset(train_data, train=True)
        validset = DiosDataset(valid_data, train=False)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2, timeout=10
        )
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=2, timeout=10
        )
        optimizer = optim.Adam(
            self.system_model.parameters(), lr=config["learning_rate"], weight_decay=0.3
        )

        train_loss_logger = LossLogger()
        valid_loss_logger = LossLogger()
        for epoch in range(config["epoch"]):
            train_loss_logger.start_epoch()
            valid_loss_logger.start_epoch()
            for i, batch in enumerate(trainloader, 0):
                optimizer.zero_grad()
                loss, loss_dict = self._compute_batch_loss(batch)
                train_loss_logger.update(loss, loss_dict)
                loss.backward()
                optimizer.step()

            for i, batch in enumerate(validloader, 0):
                loss, loss_dict = self._compute_batch_loss(batch)
                valid_loss_logger.update(loss, loss_dict)
            train_loss_logger.end_epoch()
            valid_loss_logger.end_epoch()
            print(
                "[{:4d}] ".format(epoch + 1),
                train_loss_logger.get_msg("train"),
                valid_loss_logger.get_msg("valid"),
            )
        return train_loss_logger, valid_loss_logger
