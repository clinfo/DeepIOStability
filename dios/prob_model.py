import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dios.dios_data import DiosDataset


class LossLogger:
    loss_history = []
    loss_dict_history = []

    def start_epoch(self):
        self.running_loss = 0
        self.running_loss_dict = {}

    def update(self, loss, loss_dict):
        self.running_loss += loss
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[k] += v
            else:
                self.running_loss_dict[k] = v

    def stop_epoch(self):
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


# class P_obs_given_state():
# class P_transition():
# class Q_state_given_obs():
class DiosSSMProb:
    def __init__(self, config, encoder, decoder):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def parameters(self):
        traninable_nets = ["encoder", "decoder"]
        params = []
        for net_name in traninable_nets:
            net = getattr(self, net_name)
            params.extend(net.parameters())
        return params

    def reparameterize(self, z_mean, z_var):
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(
            torch.zeros_like(z_mean), torch.ones_like(z_var)
        )
        return q_z, p_z

    def compute_batch_loss(self, batch):
        obs, input_, label = batch
        metrics = {}
        # forward + backward + optimize
        state_mean, state_var = self.encoder(obs)
        q_z, p_z = self.reparameterize(state_mean, state_var)
        state_z = q_z.rsample()
        recons_mean, recons_var = self.decoder(state_z)
        ###
        # log_p_z = p_z.log_prob(z)
        # log_q_z_x = q_z.log_prob(z)
        ###
        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z)
        loss_recons = (recons_mean - obs) ** 2
        loss_d = loss_KL.sum(dim=(1, 2)) + loss_recons.sum(dim=(1, 2))
        loss = loss_d.mean()
        metrics["loss_KL"] = loss_KL.sum(dim=(1, 2)).mean()
        metrics["loss_recons"] = loss_recons.sum(dim=(1, 2)).mean()
        return loss, metrics

    def fit(self, train_data, valid_data):
        config = self.config
        trainset = DiosDataset(train_data, train=True)
        validset = DiosDataset(train_data, train=False)
        trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
        validloader = DataLoader(validset, batch_size=10, shuffle=True, num_workers=2)
        # optimizer = optim.SGD(self.parameters(),lr=config["learning_rate"], momentum=0.99, nesterov=True,weight_decay=0.3)
        optimizer = optim.Adam(
            self.parameters(), lr=config["learning_rate"], weight_decay=0.3
        )

        for epoch in range(config["epoch"]):
            running_train_loss = 0.0
            running_valid_loss = 0.0
            for i, batch in enumerate(trainloader, 0):
                # zero the parameter gradients
                optimizer.zero_grad()
                loss, metrics = self.compute_batch_loss(batch)
                loss.backward()
                optimizer.step()

                # print statistics
                running_train_loss += loss.item()
                # print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1, i+1, running_loss/100))
            with torch.no_grad():
                for i, batch in enumerate(validloader, 0):
                    loss, metrics = self.compute_batch_loss(batch)
                    running_valid_loss += loss.item()
            print(
                "[{:d}, {:5d}] train-loss: {:.3f}  valid-loss: {:.3f}".format(
                    epoch + 1, i + 1, running_train_loss, running_valid_loss
                )
            )
            running_loss = 0.0
        print("Finished Training")
        """
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in testloader:
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))
        """
