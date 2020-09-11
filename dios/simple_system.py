import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, activation=F.relu, scale=0.1):
        super(SimpleMLP, self).__init__()
        linears=[]
        prev_d=in_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        linears.append(self.get_layer(prev_d,out_dim))
        self.linears = nn.ModuleList(linears)
        self.activation = activation
        self.scale = scale

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        nn.init.kaiming_uniform_(l.weight)
        return l

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = self.activation(self.linears[i](x))
        x = self.linears[len(self.linears)-1](x)
        return x*self.scale

class SimpleV1(torch.nn.Module):
    def __init__(self, in_dim, device=None):
        super(SimpleV1, self).__init__()
        self.in_dim = in_dim
        self.pt1 = torch.zeros((self.in_dim,))
        self.pt1=self.pt1.to(device)

    def get_stablle_points(self):
        return [self.pt1]

    def forward(self, x):
        v1 = x - self.pt1
        v1 = (v1 ** 2).sum(dim=(2,))
        vv = torch.stack([v1])
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index     

class SimpleV2(torch.nn.Module):
    def __init__(self, in_dim, device=None):
        super(SimpleV2, self).__init__()
        self.in_dim = in_dim
        self.pt1 = torch.zeros((self.in_dim,))
        self.pt2 = torch.zeros((self.in_dim,))
        self.pt1[0] = 1.0
        self.pt2[0] = -1.0
        self.pt1=self.pt1.to(device)
        self.pt2=self.pt2.to(device)

    def get_stablle_points(self):
        return [self.pt1,self.pt2]

    def forward(self, x):
        v1 = x - self.pt1
        v2 = x - self.pt2
        v1 = (v1 ** 2).sum(dim=(2,))
        v2 = (v2 ** 2).sum(dim=(2,))
        vv = torch.stack([v1, v2])
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index

class SimpleV3(torch.nn.Module):
    def __init__(self, in_dim, device=None):
        super(SimpleV3, self).__init__()
        self.in_dim = in_dim
        self.pts=[]
        for i in range(in_dim):
            pt1 = torch.zeros((self.in_dim,))
            pt2 = torch.zeros((self.in_dim,))
            pt1[i] = 1.0
            pt2[i] = -1.0
            pt1=pt1.to(device)
            pt2=pt2.to(device)
            self.pts.append(pt1)
            self.pts.append(pt2)

    def get_stablle_points(self):
        return self.pts

    def forward(self, x):
        vv=[]
        for pt in self.pts:
            v1 = x - pt
            v1 = (v1 ** 2).sum(dim=(2,))
            vv.append(v1)
        vv = torch.stack(vv)
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index



class SimpleSystem(torch.nn.Module):
    def __init__(
        self,
        obs_dim,
        state_dim,
        input_dim,
        delta_t=0.1,
        gamma=None,
        c=0.1,
        init_state_mode="estimate_state",
        alpha=[1.0,1.0,1.0,1.0],
        hidden_layer_h=None,
        hidden_layer_f=None,
        hidden_layer_g=None,
        diag_g=True,
        scale=0.1,
        v_type="single",
        device=None,
    ):
        super(SimpleSystem, self).__init__()
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.delta_t = delta_t
        self.c=c
        self.init_state_mode=init_state_mode
        self.alpha=alpha
        self.diag_g=diag_g
        self.device=device

        self.state_dim = state_dim
        self.input_dim = input_dim

        self.func_h = SimpleMLP(state_dim, hidden_layer_h, obs_dim,scale=scale)
        self.func_h_inv = SimpleMLP(obs_dim, hidden_layer_h, state_dim,scale=scale)
        self.func_f = SimpleMLP(state_dim, hidden_layer_f, state_dim,scale=scale)
        self.func_g = SimpleMLP(state_dim, hidden_layer_g, state_dim * input_dim,scale=scale)
        self.func_g_vec = SimpleMLP(state_dim, hidden_layer_g, input_dim,scale=scale)
        if v_type=="single":
            self.func_v = SimpleV1(state_dim,device=device)
        elif v_type=="double":
            self.func_v = SimpleV2(state_dim,device=device)
        elif v_type=="many":
            self.func_v = SimpleV3(state_dim,device=device)
        else:
            print("[ERROR] unknown:",v_type)

        self._input_state_eye = torch.eye(self.input_dim, self.state_dim,device=device)
        self.param_gamma = nn.Parameter(torch.randn(1)[0]+5)

    def func_g_mat(self, x):
        if self.diag_g:
            if len(x.shape) == 2:  # batch_size x state_dim
                g = self.func_g_vec(x)
                temp = self._input_state_eye
                temp = temp.reshape((1, self.input_dim, self.state_dim))
                temp = temp.repeat(x.shape[0], 1, 1)
                y = temp*g.reshape((x.shape[0], self.input_dim, 1))
            elif len(x.shape) == 3:  # batch_size x time x state_dim
                g = self.func_g_vec(x)
                temp = self._input_state_eye
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
        v,i_v = self.func_v(x)
        # v is independet w.r.t. batch and time
        dv = torch.autograd.grad(v.sum(), x, create_graph=True)[0]

        hj_dvf = torch.sum(dv * self.func_f(x), dim=-1)
        h = self.func_h(x)
        if self.gamma is None:
            gamma = (1.0e-4+F.relu(self.param_gamma))
        else:
            gamma = self.gamma
        ##
        mu_list=self.func_v.get_stablle_points()
        hj_hh_list=[]
        h = self.func_h(x)
        for mu in mu_list:
            h_mu = self.func_h(mu)
            hh= 1 / 2.0 * torch.sum((h-h_mu) * (h-h_mu), dim=-1)
            hj_hh_list.append(hh)
        if len(mu_list)==2:
            hj_hh=torch.where(i_v==1, hj_hh_list[1],hj_hh_list[0])
        elif len(mu_list)==1:
            hj_hh=hj_hh_list[0]
        elif len(mu_list)>2:
            hj_hh=hj_hh_list[0]
            for i in range(len(mu_list)-1):
                hj_hh=torch.where(i_v==i+1, hj_hh_list[i+1],hj_hh)
        ##

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
        if type(self.init_state_mode) is str:
            if   self.init_state_mode=="true_state":
                init_state =state[:,0,:]
            elif self.init_state_mode=="random_state":
                init_state =torch.randn(*(batch_size, self.state_dim),device=self.device)
            elif self.init_state_mode=="estimate_state":
                init_state = self.func_h_inv(obs[:, 0, :])
            elif self.init_state_mode=="zero_state":
                init_state =torch.zeros((batch_size, self.state_dim),device=self.device)
            else:
                print("[ERROR] unknown init_state:",state.init_state_mode)
        else:
            init_state =torch.zeros((batch_size, self.state_dim),device=self.device)
            init_state+=self.init_state_mode
        state_generated, obs_generated = self.simutlate(init_state, batch_size, step, input_)
        return state_generated, obs_generated

    def forward_state_sampling(self,n,m,scale):
        mu_list=self.func_v.get_stablle_points()
        ret=[]
        for mu in mu_list:
            x= scale*torch.randn((n,m,self.state_dim),requires_grad=True,device=self.device)
            ret.append(x)
        ret=torch.cat(ret, dim=0)
        return ret

    def forward_loss(self, obs, input_, state, obs_generated, state_generated, with_state_loss=False, epoch=None):
        ### observation loss
        # print("obs(r)",obs_generated.shape)
        # print("obs",obs.shape)
        # print("state",state.shape)
        loss_recons = (obs - obs_generated) ** 2
        #state_rand =3*torch.randn(state.shape,requires_grad=True,device=self.device)
        state_rand =self.forward_state_sampling(10,10,scale=1.0)
        loss_hj, loss_hj_list = self.compute_HJ(state_rand)
        #loss_hj, loss_hj_list = self.compute_HJ(state)
        loss_hj = F.relu(loss_hj + self.c)
        step_wise_loss=False
        if step_wise_loss:
            loss_sum_recons=loss_recons.sum(dim=2).mean(dim=(0,1))
            loss_sum_hj=loss_hj.mean(dim=(0,1))
        else:
            loss_sum_recons=loss_recons.sum(dim=(1,2)).mean(dim=0)
            loss_sum_hj=loss_hj.sum(dim=1).mean(dim=0)
        if epoch is not None and epoch<3:
            loss = {
                "recons": self.alpha[0]*loss_sum_recons,
                "*recons": loss_sum_recons,
            }
        else:
            loss = {
                "recons": self.alpha[0]*loss_sum_recons,
                "HJ": self.alpha[1]*loss_sum_hj,
                "*recons": loss_sum_recons,
                "*HJ": loss_sum_hj,
                "*HJ_vf": loss_hj_list[0].sum(dim=1).mean(dim=0),
                "*HJ_hh": loss_hj_list[1].sum(dim=1).mean(dim=0),
                "*HJ_gg": loss_hj_list[2].sum(dim=1).mean(dim=0),
            }
        if self.gamma is None:
            loss["gamma"]=self.alpha[2]*self.param_gamma**2
            loss["*gamma"]=self.param_gamma**2
        if with_state_loss:
            if state is not None:
                loss_state = (state - state_generated) ** 2
                loss_sum_state = loss_state.sum(dim=(1,2)).mean(dim=0)
                loss["*state"] = self.alpha[3] * loss_sum_state
        return loss

    def forward(self, obs, input_, state=None, with_generated=False, epoch=None):
        state_generated, obs_generated = self.forward_simulate(obs, input_, state)
        l2_gain=self.forward_l2gain(obs, input_)
        l2_gain_recons=self.forward_l2gain(obs_generated, input_)
        loss=self.forward_loss(obs, input_, state_generated, obs_generated, state, epoch=epoch)
        loss["*l2"]=l2_gain.mean()
        loss["*l2_recons"]=l2_gain_recons.mean()
        if with_generated:
            return loss, state_generated, obs_generated
        else:
            return loss
