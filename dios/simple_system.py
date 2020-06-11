import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, activation=F.relu):
        super(SimpleMLP, self).__init__()
        self.fc_0 = nn.Linear(in_dim, h_dim)
        self.fc_1 = nn.Linear(h_dim, h_dim)
        self.fc_2 = nn.Linear(h_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc_0(x))
        x = self.activation(self.fc_1(x))
        x = self.fc_2(x)
        return x


class SimpleV(torch.nn.Module):
    def __init__(self, in_dim):
        super(SimpleV, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pt1 = torch.zeros((self.in_dim,))
        pt2 = torch.zeros((self.in_dim,))
        pt1[0] = 1.0
        pt2[0] = -1.0
        v1 = x - pt1
        v2 = x - pt2
        v1 = (v1 ** 2).sum(dim=(2,))
        v2 = (v2 ** 2).sum(dim=(2,))
        vv = torch.stack([v1, v2])
        min_vv, _ = torch.min(vv, dim=0)
        return min_vv


class SimpleSystem(torch.nn.Module):
    def __init__(
        self,
        obs_dim,
        state_dim,
        input_dim,
        delta_t=0.1,
        gamma=1.0,
        c=0.1,
        init_state_mode="estimate_state",
        alpha=[1.0,1.0,1.0],
        hidden_layer_dim=None,
        diag_g=True,
    ):
        super(SimpleSystem, self).__init__()
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.delta_t = delta_t
        self.c=c
        self.init_state_mode=init_state_mode
        self.alpha=alpha
        self.diag_g=diag_g

        self.state_dim = state_dim
        self.input_dim = input_dim
        if hidden_layer_dim is None:
            hidden_layer_dim = state_dim if state_dim > obs_dim else obs_dim

        self.func_h = SimpleMLP(state_dim, hidden_layer_dim, obs_dim)
        self.func_f = SimpleMLP(state_dim, hidden_layer_dim, state_dim)
        self.func_h_inv = SimpleMLP(obs_dim, hidden_layer_dim, state_dim)
        self.func_g = SimpleMLP(state_dim, hidden_layer_dim, state_dim * input_dim)
        self.func_g_vec = SimpleMLP(state_dim, hidden_layer_dim, input_dim)
        self.func_v = SimpleV(state_dim)

    def func_g_mat(self, x):
        if self.diag_g:
            if len(x.shape) == 2:  # batch_size x state_dim
                g = self.func_g_vec(x)
                temp = torch.eye(self.input_dim, self.state_dim)
                temp = temp.reshape((1, self.input_dim, self.state_dim))
                temp = temp.repeat(x.shape[0], 1, 1)
                y = temp*g.reshape((x.shape[0], self.input_dim, 1))
            elif len(x.shape) == 3:  # batch_size x time x state_dim
                g = self.func_g_vec(x)
                temp = torch.eye(self.input_dim, self.state_dim)
                temp = temp.reshape((1, 1, self.input_dim, self.state_dim))
                temp = temp.repeat(x.shape[0], x.shape[1], 1, 1)
                y = temp*g.reshape((x.shape[0], x.shape[1], self.input_dim, 1))
            else:
                print("error", x.shape)
            return y
        else:
            if len(x.shape) == 2:  # batch_size x state_dim
                g = self.func_g(x).reshape((x.shape[0], self.input_dim, self.state_dim))
            elif len(x.shape) == 3:  # batch_size x time x state_dim
                g = self.func_g(x).reshape(
                    (x.shape[0], x.shape[1], self.input_dim, self.state_dim)
                )
            else:
                print("error", x.shape)
            return g

    def compute_HJ(self, x):
        v = self.func_v(x)
        # v is independet w.r.t. batch and time
        dv = torch.autograd.grad(v.sum(), x, create_graph=True)[0]

        hj_dvf = torch.sum(dv * self.func_f(x), dim=-1)
        h = self.func_h(x)
        gamma = self.gamma
        hj_hh = 1 / 2.0 * torch.sum(h * h, dim=-1)
        g = self.func_g_mat(x)
        dv = torch.unsqueeze(dv, -1)
        gdv = torch.matmul(g, dv)
        gdv2 = torch.sum(gdv ** 2, dim=(-1, -2))
        hj_gg = 1 / (2.0 * gamma ** 2) * gdv2

        loss_hj = hj_dvf + hj_hh + hj_gg
        return loss_hj, [hj_dvf, hj_hh, hj_gg]

    def vecmat_mul(self, vec, mat):
        vec_m = torch.unsqueeze(vec, -2)
        o = torch.matmul(vec_m, mat)
        o = torch.squeeze(o, -2)
        return o

    def simutlate(self, init_state, batch_size, step, input_=None):
        current_state = init_state
        state = []
        obs_generated = []
        for t in range(step):
            ## simulating system
            if input_ is not None:
                ###
                g = self.func_g_mat(current_state)
                u = input_[:, t, :]
                ug = self.vecmat_mul(u, g)
                ###
                next_state = (
                    current_state + (self.func_f(current_state) + ug) * self.delta_t
                )
            else:
                next_state = current_state + self.func_f(current_state) * self.delta_t
            o = self.func_h(current_state)
            ## saving time-point data
            obs_generated.append(o)
            state.append(current_state)
            current_state = next_state
        obs_generated = torch.stack(obs_generated, dim=1)
        state = torch.stack(state, dim=1)
        return state, obs_generated

    def forward_l2gain(self, obs_generated, input_):
        # batch x time x dimension
        y=(obs_generated**2).sum(dim=2)
        u=(input_**2).sum(dim=2)
        y_shift=torch.roll(y, 1, dims=1)
        u_shift=torch.roll(u, 1, dims=1)
        y_l2=((y_shift[:,1:]+y[:,:-1])* self.delta_t/2).sum(dim=1)
        u_l2=((u_shift[:,1:]+u[:,:-1])* self.delta_t/2).sum(dim=1)
        return y_l2/(u_l2+1.0e-10)

    def forward_simulate(self, obs, input_, state=None):
        # obs=torch.tensor(obs, requires_grad=True)
        step = obs.shape[1]
        batch_size = obs.shape[0]
        if   self.init_state_mode=="true_state":
            init_state=state[:,0,:]
        elif self.init_state_mode=="random_state":
            init_state=torch.randn(*(batch_size, self.state_dim))
        elif self.init_state_mode=="estimate_state":
            init_state = self.func_h_inv(obs[:, 0, :])
        else:
            print("[ERROR] unknown init_state:",state.init_state_mode)
        init_state = self.func_h_inv(obs[:, 0, :])
        state_generated, obs_generated = self.simutlate(init_state, batch_size, step, input_)
        return state_generated, obs_generated

    def forward_loss(self, obs, input_, state, obs_generated, state_generated, with_state_loss=False):
        ### observation loss
        # print("obs(r)",obs_generated.shape)
        # print("obs",obs.shape)
        # print("state",state.shape)
        loss_recons = self.alpha[0]*(obs - obs_generated) ** 2
        loss_hj, loss_hj_list = self.compute_HJ(state)
        loss_hj = self.alpha[1]*F.relu(loss_hj + self.c)
        """
        loss = {
            "recons": loss_recons.sum(),
            "HJ": loss_hj.sum(),
            "*HJ_vf": loss_hj_list[0].sum(),
            "*HJ_hh": loss_hj_list[1].sum(),
            "*HJ_gg": loss_hj_list[2].sum(),
            #"recons": loss_recons.sum(dim=(1,2)).mean(dim=0),
            #"HJ": loss_hj.sum(dim=1).mean(dim=0),
        }
        """
        loss = {
            "recons": loss_recons.sum(dim=(1,2)).mean(dim=0),
            "HJ": loss_hj.sum(dim=1).mean(dim=0),
            "*HJ_vf": loss_hj_list[0].sum(dim=1).mean(dim=0),
            "*HJ_hh": loss_hj_list[1].sum(dim=1).mean(dim=0),
            "*HJ_gg": loss_hj_list[2].sum(dim=1).mean(dim=0),
            #"recons": loss_recons.sum(dim=(1,2)).mean(dim=0),
            #"HJ": loss_hj.sum(dim=1).mean(dim=0),
        }
        if with_state_loss:
            if state is not None:
                loss_state = self.alpha[2] * (state - state_generated) ** 2
                loss["state"]=loss_state.sum(dim=(1,2)).mean(dim=0)
        return loss

    def forward(self, obs, input_, state=None, with_generated=False):
        state_generated, obs_generated = self.forward_simulate(obs, input_, state)
        l2_gain=self.forward_l2gain(obs, input_)
        l2_gain_recons=self.forward_l2gain(obs_generated, input_)
        loss=self.forward_loss(obs, input_, state_generated, obs_generated, state)
        loss["*l2"]=l2_gain.mean()
        loss["*l2_recons"]=l2_gain_recons.mean()
        if with_generated:
            return loss, state_generated, obs_generated
        else:
            return loss
