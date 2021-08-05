import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dios.data_util import DiosDataset
import json
import logging
import numpy as np


class LossLogger:
    def __init__(self):
        self.loss_history=[]
        self.loss_dict_history=[]

    def start_epoch(self):
        self.running_loss = 0
        self.running_loss_dict = {}
        self.running_count = 0

    def update(self, loss, loss_dict):
        self.running_loss += loss.detach().to('cpu')
        self.running_count +=1
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[k] += v.detach().to('cpu')
            else:
                self.running_loss_dict[k] = v.detach().to('cpu')

    def end_epoch(self,mean_flag=True):
        if mean_flag:
            self.running_loss /= self.running_count
            for k in self.running_loss_dict.keys():
                self.running_loss_dict[k] /=  self.running_count
        self.loss_history.append(self.running_loss)
        self.loss_dict_history.append(self.running_loss_dict)

    def get_dict(self, prefix="train"):
        result={}
        key="{:s}-loss".format(prefix)
        val=self.running_loss
        result[key]=float(val)
        for k, v in self.running_loss_dict.items():
            if k[0]!="*":
                m = "{:s}-{:s}-loss".format(prefix, k)
            else:
                m = "*{:s}-{:s}".format(prefix, k[1:])
            result[m]=float(v)
        return result

    def get_msg(self, prefix="train"):
        msg = []
        for key,val in self.get_dict(prefix=prefix).items():
            m = "{:s}: {:.3f}".format(key,val)
            msg.append(m)
        return "  ".join(msg)

    def get_loss(self):
            return self.running_loss

class DiosSSM:
    def __init__(self, config, system_model, device):
        self.config = config
        self.system_model = system_model.to(device)
        self.device=device
        self.logger = logging.getLogger("logger")

    def _simulate_with_batch(self, batch, step_wise_loss=False):
        obs, input_, state = None, None, None
        obs=batch["obs"]
        if "input" in batch:
            input_ = batch["input"]
        if "state" in batch:
            state = batch["state"]
        loss_dict, state_generated, obs_generated, hh = self.system_model.forward(obs, input_, state, with_generated=True,step_wise_loss=step_wise_loss)
        loss = 0
        for k, v in loss_dict.items():
            if k[0]!="*":
                loss += v
        return loss, loss_dict, state_generated, obs_generated, hh

    def _compute_batch_loss(self, batch, epoch):
        obs, input_, state = None, None, None
        obs=batch["obs"]
        if "input" in batch:
            input_ = batch["input"]
        if "state" in batch:
            state = batch["state"]
        loss_dict = self.system_model(obs, input_, state, epoch=epoch)
        loss = 0
        for k, v in loss_dict.items():
            if k[0]!="*":
                loss += v
        return loss, loss_dict


    def get_vector_field(self, state_dim, dim=[0,1],min_v=-3,max_v=3,delta=0.5):
        if state_dim==1:
            x = np.arange(min_v, max_v, delta)
            state= np.reshape(x,(-1,1))
        else:
            x1 = np.arange(min_v, max_v, delta)
            x2 = np.arange(min_v, max_v, delta)
            xx = np.meshgrid(x1,x2)
            np.array(xx).shape
            vv=np.transpose(np.reshape(np.array(xx),(2,-1)))
            state=np.zeros((vv.shape[0],state_dim))
            state[:,dim]=vv
        state_t=torch.tensor(state,dtype=torch.float32,device=self.device)
        next_state = self.system_model.simulate_one_step(state_t)
        vec = next_state-state_t
        return state, vec.detach().to('cpu')

    def simulate_with_input(self, input_data, init_state, step=100):
        if input_data is not None:
            input_data=input_data.to(self.device)
        init_state=init_state.to(self.device)
        state, obs_generated=self.system_model.simulate(init_state, batch_size=init_state.shape[0], step=step, input_=input_data)
        return state.detach().to('cpu'),obs_generated.detach().to('cpu')


    def simulate_with_data(self, valid_data, step_wise_loss=False):
        config = self.config
        validset = DiosDataset(valid_data, train=False)
        batch_size = config["batch_size"]
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20
        )

        valid_loss_logger = LossLogger()
        valid_loss_logger.start_epoch()
        state_generated_list, obs_generated_list = [], []
        hh_list=[]
        for i, batch in enumerate(validloader, 0):
            batch={k:el.to(self.device) for k,el in batch.items()}
            loss, loss_dict, state_generated, obs_generated, hh = self._simulate_with_batch(batch,step_wise_loss=step_wise_loss)
            state_generated_list.append(state_generated)
            obs_generated_list.append(obs_generated)
            hh_list.append(hh)
            valid_loss_logger.update(loss, loss_dict)
            del loss
            del loss_dict
        valid_loss_logger.end_epoch()

        print(valid_loss_logger.get_msg("valid"))
        out_state_generated = torch.cat(state_generated_list, dim=0)
        out_obs_generated = torch.cat(obs_generated_list, dim=0)
        out_hh = torch.cat(hh_list, dim=0)
        return valid_loss_logger, out_state_generated, out_obs_generated, out_hh

    def save(self,path):
        self.logger.info("[save model]"+path)
        torch.save(self.system_model.state_dict(), path)

    def load(self,path):
        self.logger.info("[load model]"+path)
        state_dict=torch.load(path)
        self.system_model.load_state_dict(state_dict)
        self.system_model.eval()

    def save_ckpt(self, epoch, loss, optimizer, path):
        self.logger.info("[save ckeck point]"+path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.system_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

    def load_ckpt(self, path):
        self.logger.info("[load ckeck point]"+path)
        ckpt=torch.load(path)
        self.system_model.load_state_dict(ckpt["model_state_dict"])
        self.system_model.eval()

    def fit(self, train_data, valid_data):
        config = self.config
        batch_size = config["batch_size"]
        trainset = DiosDataset(train_data, train=True)
        validset = DiosDataset(valid_data, train=False)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, timeout=20
        )
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20
        )
        """
        optimizer = optim.Adam(
            self.system_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        """
        optimizer = optim.RMSprop(
            self.system_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
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
                batch={k:el.to(self.device) for k,el in batch.items()}
                loss, loss_dict = self._compute_batch_loss(batch,epoch)
                train_loss_logger.update(loss, loss_dict)
                loss.backward()
                # grad clipping by norm
                torch.nn.utils.clip_grad_norm_(self.system_model.parameters(), max_norm=1.0, norm_type=2)
                # grad clipping by value
                #torch.nn.utils.clip_grad_norm_(self.system_model.parameters(), 1.0e-1)
                optimizer.step()
                del loss
                del loss_dict

            for i, batch in enumerate(validloader, 0):
                #batch=[el.to(self.device) for el in batch]
                batch={k:el.to(self.device) for k,el in batch.items()}
                loss, loss_dict = self._compute_batch_loss(batch,epoch)
                valid_loss_logger.update(loss, loss_dict)
            train_loss_logger.end_epoch()
            valid_loss_logger.end_epoch()
            ## Early stopping
            l=valid_loss_logger.get_loss()
            if np.isnan(l):
                self.logger.info("... nan is detected in training")
                msg="\t".join(["[{:4d}] ".format(epoch + 1),
                    train_loss_logger.get_msg("train"),
                    valid_loss_logger.get_msg("valid"),
                    "({:2d})".format(patient_count),])
                self.logger.info(msg)
                return train_loss_logger, valid_loss_logger, False
            elif epoch >20 and l>1.0e15:
                self.logger.info("... loss is too learge")
                msg="\t".join(["[{:4d}] ".format(epoch + 1),
                    train_loss_logger.get_msg("train"),
                    valid_loss_logger.get_msg("valid"),
                    "({:2d})".format(patient_count),])
                self.logger.info(msg)
                return train_loss_logger, valid_loss_logger, False
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
                path = config["save_model_path"]+f"/best.result.json"
                fp = open(path, "w")
                res=train_loss_logger.get_dict("train")
                res.update(valid_loss_logger.get_dict("valid"))
                json.dump(
                    res,
                    fp,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                check_point_flag=True
                best_valid_loss=l

            ## print message
            ckpt_msg = "*" if check_point_flag else ""
            msg="\t".join(["[{:4d}] ".format(epoch + 1),
                train_loss_logger.get_msg("train"),
                valid_loss_logger.get_msg("valid"),
                "({:2d})".format(patient_count),
                ckpt_msg,])
            self.logger.info(msg)
        return train_loss_logger, valid_loss_logger, True
