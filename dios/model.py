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

    def end_epoch(self,mean_flag=True):
        if mean_flag:
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
            if k[0]!="*":
                m = "{:s}-{:s}-loss: {:.3f}".format(prefix, k, v)
            else:
                m = "*{:s}-{:s}: {:.3f}".format(prefix, k[1:], v)
            msg.append(m)
        return "  ".join(msg)

    def get_loss(self):
            return self.running_loss

class DiosSSM:
    def __init__(self, config, system_model):
        self.config = config
        self.system_model = system_model

    def _compute_batch_simulate(self, batch):
        obs, input_, state = batch
        metrics = {}
        if input_ is 0:  ## to avoid error (specification of pytorch)
            input_ = None

        loss_dict, state_generated, obs_generated = self.system_model.forward(obs, input_, state, with_generated=True)
        loss = 0
        for k, v in loss_dict.items():
            if k[0]!="*":
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
            if k[0]!="*":
                loss += v
        return loss, loss_dict

    def save(self,path):
        torch.save(self.system_model.state_dict(), path)

    def load(self,path):
        state_dict=torch.load(path)
        self.system_model.load_state_dict(state_dict)
        self.system_model.eval()

    def save_ckpt(self, epoch, loss, optimizer, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.system_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

    def load_ckpt(self, path):
        ckpt=torch.load(path)
        self.system_model.load_state_dict(ckpt["model_state_dict"])
        self.system_model.eval()

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
        prev_valid_loss=None
        best_valid_loss=None
        patient_count=0
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
            ## Early stopping
            l=valid_loss_logger.get_loss()
            if prev_valid_loss is None or l < prev_valid_loss:
                patient_count=0
            else:
                patient_count+=1
            prev_valid_loss=l
            ## check point
            check_point_flag=False
            if best_valid_loss is None or l < best_valid_loss:
                path = config["save_model_path"]+f"/model.{epoch}.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                path = config["save_model_path"]+f"/best.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                check_point_flag=True
                best_valid_loss=l

            ## print message
            ckpt_msg = "*" if check_point_flag else ""
            print(
                "[{:4d}] ".format(epoch + 1),
                train_loss_logger.get_msg("train"),
                valid_loss_logger.get_msg("valid"),
                "({:2d})".format(patient_count),
                ckpt_msg,
            )
        return train_loss_logger, valid_loss_logger
