import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import math

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, activation=F.leaky_relu, scale=0.1, residual=False, residual_coeff=-1.0, offset=0.0, positive=False, with_bn=False,with_dropout=False):
        super(SimpleMLP, self).__init__()
        linears=[]
        bns=[]
        dos=[]
        self.with_bn=with_bn
        self.with_dropout=with_dropout
        prev_d=in_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            if with_bn:
                bns.append(nn.BatchNorm1d(d))
            if with_dropout:
                dos.append(nn.Dropout(p=0.3))
            prev_d=d
        linears.append(self.get_layer(prev_d,out_dim))
        self.linears = nn.ModuleList(linears)
        self.bns = nn.ModuleList(bns)
        self.dos = nn.ModuleList(dos)
        self.activation = activation
        self.scale = scale
        self.residual = residual
        self.residual_coeff = residual_coeff
        self.offset = offset
        self.positive = positive

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l

    def forward(self, x):
        res_x=x
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            if self.with_bn:
                x = self.bns[i](x)
            if self.with_dropout:
                x = self.dos[i](x)
            x = self.activation(x)
        x = self.linears[len(self.linears)-1](x)
        if self.residual:
            out = x*self.scale+self.residual_coeff*res_x+self.offset
        else:
            out = x*self.scale+self.offset
        if self.positive:
            return 1.0e-4+F.relu(out)
        else:
            return out

class SimpleV1(torch.nn.Module):
    def __init__(self, in_dim, device=None):
        super(SimpleV1, self).__init__()
        self.in_dim = in_dim
        self.pt1 = torch.zeros((self.in_dim,))
        self.pt1=self.pt1.to(device)

    def get_stable_points(self):
        return [self.pt1]
    def get_limit_cycles(self):
        return []
    def get_stable(self):
        return [("point",self.pt1)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        v1 = x - self.pt1
        v1 = (v1 ** 2).sum(dim=(-1,))
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

    def get_stable_points(self):
        return [self.pt1,self.pt2]
    def get_limit_cycles(self):
        return []
    def get_stable(self):
        return [("point",self.pt1),("point",self.pt2)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        v1 = x - self.pt1
        v2 = x - self.pt2
        v1 = (v1 ** 2).sum(dim=(-1,))
        v2 = (v2 ** 2).sum(dim=(-1,))
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

    def get_stable_points(self):
        return self.pts
    def get_limit_cycles(self):
        return []
    def get_stable(self):
        return [("point",pt) for pt in self.pts]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        vv=[]
        for pt in self.pts:
            v1 = x - pt
            v1 = (v1 ** 2).sum(dim=(-1,))
            vv.append(v1)
        vv = torch.stack(vv)
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index

## limit cycle
class SimpleV4(torch.nn.Module):
    def __init__(self, in_dim, device=None):
        super(SimpleV4, self).__init__()
        self.in_dim = in_dim
        self.device = device
        self.pt1 = torch.zeros((self.in_dim,),device=device)

    def get_stable_points(self):
        return []
    def get_limit_cycles(self):
        return [self.pt1]
    def get_stable(self):
        return [("limit_cycle",self.pt1)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        eps=1.0e-4
        nx=x/(torch.sqrt(torch.sum(x**2,dim=-1,keepdim=True)+eps)+eps)
        d=torch.sum((x-nx)**2,dim=-1)
        min_i=torch.zeros(d.size(),device=self.device)
        return d,min_i





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
        alpha={},
        hj_loss_type="const",
        hidden_layer_h=None,
        hidden_layer_f=None,
        hidden_layer_g=None,
        diag_g=True,
        stable_type="fgh",# none/f/fg/fgh
        scale=0.1,
        v_type="single",
        init_gamma=5,
        system_f_residual=True,
        system_f_residual_coeff=-1,
        system_g_offset=0,
        system_g_positive=False,
        schedule_pretrain_epoch=3,
        device=None,
    ):
        super(SimpleSystem, self).__init__()
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.delta_t = delta_t
        self.c=c
        self.init_state_mode=init_state_mode
        self.alpha={"recons":1.0,"HJ":1.0,"gamma":1.0,"state":1.0,"HJ_dvf":1.0,"HJ_hh":1.0,"HJ_gg":1.0,}
        self.alpha.update(alpha)
        self.schedule_alpha=alpha
        self.diag_g=diag_g
        self.device=device

        self.state_dim = state_dim
        self.input_dim = input_dim
        
        self.hj_loss_type=hj_loss_type
    
        if state_dim == obs_dim:
            self._func_h = SimpleMLP(state_dim, hidden_layer_h, obs_dim,
                    scale=scale, with_bn=False, residual=True)
        else:
            self._func_h = SimpleMLP(state_dim, hidden_layer_h, obs_dim,
                    scale=scale, with_bn=False, residual=False)
        self._func_h_inv = SimpleMLP(obs_dim, hidden_layer_h, state_dim,
                scale=scale, with_bn=False)
        self._func_f = SimpleMLP(state_dim, hidden_layer_f, state_dim,
                scale=1.0, residual=system_f_residual,
                residual_coeff=system_f_residual_coeff,with_bn=False)
        self._func_g = SimpleMLP(state_dim, hidden_layer_g, state_dim * input_dim,
                scale=1.0,with_bn=False)
        self._func_g_vec = SimpleMLP(state_dim, hidden_layer_g, input_dim,
                scale=1.0,with_bn=False)
        if v_type=="single":
            self.func_v = SimpleV1(state_dim,device=device)
        elif v_type=="double":
            self.func_v = SimpleV2(state_dim,device=device)
        elif v_type=="many":
            self.func_v = SimpleV3(state_dim,device=device)
        elif v_type=="single_cycle":
            self.func_v = SimpleV4(state_dim,device=device)
        else:
            print("[ERROR] unknown:",v_type)

        self._input_state_eye = torch.eye(self.input_dim, self.state_dim,device=device)
        self.param_gamma = nn.Parameter(torch.randn(1)[0]+init_gamma)
        self.stable_type=stable_type
        self.schedule_stable_type=stable_type
        self.schedule_pretrain_epoch=schedule_pretrain_epoch

        self.max_value_clipping=False
        self.max_state_value=10
        self.max_obs_value=10


    def func_h(self,x):
        if len(x.shape) == 3:  # batch_size x time x state_dim
            t=x.shape[1]
            x=x.reshape((-1, self.state_dim))
            x=self._func_h(x)
            x=x.reshape((-1, t, self.obs_dim))
            return x
        else:
            return self._func_h(x)

    def func_h_inv(self,x):
        if len(x.shape) == 3:  # batch_size x time x obs_dim
            t=x.shape[1]
            x=x.reshape((-1, self.obs_dim))
            x=self._func_h_inv(x)
            x=x.reshape((-1, t, self.state_dim))
            return x
        else:
            return self._func_h_inv(x)

    def func_f(self,x):
        if len(x.shape) == 3:  # batch_size x time x state_dim
            t=x.shape[1]
            x=x.reshape((-1, self.state_dim))
            x=self._func_f(x)
            x=x.reshape((-1, t, self.state_dim))
            return x
        else:
            return self._func_f(x)

    def func_g(self,x):
        if len(x.shape) == 3:  # batch_size x time x state_dim
            t=x.shape[1]
            x=x.reshape((-1, self.state_dim))
            x=self._func_g(x)
            x=x.reshape((-1, t, self.state_dim * self.input_dim))
            return x
        else:
            return self._func_g(x)

    def func_g_vec(self,x):
        if len(x.shape) == 3:  # batch_size x time x state_dim
            t=x.shape[1]
            x=x.reshape((-1, self.state_dim))
            x=self._func_g_vec(x)
            x=x.reshape((-1, t, self.input_dim))
            return x
        else:
            return self._func_g_vec(x)

    def func_g_mat(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            g(x) (tensor): batch  (x time) x input dimension x state dimension
        """
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

    def get_gamma(self):
        if self.gamma is None:
            return self.param_gamma
        else:
            return self.gamma

    def get_stable(self):
        mu_list=self.func_v.get_stable()
        h_mu_list=[]
        for mu_type, mu in mu_list:
            if mu_type=="point":
                h_mu = self.compute_h(mu)
                h_mu_list.append(h_mu)
            else:
                h_mu_list.append(None)
        return h_mu_list, mu_list

    def get_limit_cycle(self):
        cyc_list=self.func_v.get_limit_cycles()
        #h_cyc_list=[]
        #for mu in mu_list:
        #    h_mu = self.compute_h(mu)
        #    h_mu_list.append(h_mu)
        return None, cyc_list
    

    def _unit_limit_cycle_distance(self,hx,n_sample=100, with_nearest_point=False):
        """
        Args:
            hx (tensor): batch x time x observation dimension
            n_sample (int): the number of samples
        Returns:
            squared_distances (tensor): batch x time
            min || hx - h(mu) ||^2
              mu ~ (r*cos(theta), r*sin(theta))
              theta ~ -pi - pi
              r=1

            nearest_points (tensor): batch x time x state_dim
        """
        # random sampling from a unit circle
        theta=(torch.rand((n_sample,),device=self.device)-0.5)*2*math.pi
        r=1
        mu1= r*torch.cos(theta)
        mu2= r*torch.sin(theta)
        mu = torch.stack([mu1,mu2]).T
        # mu: n_sample x 2
        if self.state_dim>2:
            pad=torch.zeros((n_sample,self.state_dim-2),device=self.device)
            mu=torch.cat([mu,pad],dim=-1)
        hmu = self.func_h(mu)
        # hmu: n_sample x observation dimension
        d_list=[]
        for i in range(n_sample):
            d=torch.sum((hx-hmu[i])**2,dim=-1)
            d_list.append(d)
        dd=torch.stack(d_list)
        min_d,min_i=torch.min(dd,dim=0)
        if with_nearest_point:
            return min_d, hmu[min_i,:]
        else:
            return min_d

    def _distance(self,hx,mu, with_nearest_point=False):
        """
        Args:
            hx (tensor): batch x time x observation dimension
            mu (tensor): batch x time x state dimension
        Returns:
            squared_distances (tensor): batch x time
            nearest_points (tensor): batch x time x state_dim
        """
        h_mu = self.func_h(mu)
        d=torch.sum((hx-h_mu)**2, dim=-1)
        if with_nearest_point:
            return d, h_mu
        else:
            return d

    def compute_Pdv(self, dv):
        """
        Args:
            dv (tensor): batch (x time) x state dimension
        Returns:
            Pdv (tensor): batch (x time) x state dimension x state dimension 
        """
        dv_mat = torch.unsqueeze(dv, -1)
        #dv_mat: batch (x time) x state dimension x 1
        Pdv = torch.matmul(dv_mat,torch.transpose(dv_mat,-1,-2))
        scale=torch.sum(dv**2,dim=-1)
        scale=torch.unsqueeze(torch.unsqueeze(scale,dim=-1),dim=-1)
        return Pdv/(scale+1.0e-10)

    def compute_scale_fgh(self, x):
        #x_=torch.tensor(x,requires_grad=True)
        x_=x.clone().detach().requires_grad_(True)
        v,i_v = self.func_v(x_)
        # v is independet w.r.t. batch and time
        dv = torch.autograd.grad(v.sum(), x_, create_graph=True)[0].detach()
        hj_dvf = torch.sum(dv * self.func_f(x_), dim=-1)
        ## gamma
        if self.gamma is None:
            gamma = (1.0e-4+F.relu(self.param_gamma))
        else:
            gamma = self.gamma
        ## hh
        hj_hh = 1 / 2.0 * self.distance_h(x, i_v)
        ## gg
        gx = self.func_g_mat(x)
        gdv2 = self.compute_GdV(dv, gx)
        hj_gg = 1 / (2.0 * gamma ** 2 + 1.0e-10) * gdv2
        return hj_dvf, hj_gg, hj_hh, dv

    def compute_f(self, x):
        """
        Args:
            x (tensor): batch x time x state dimension
        Returns:
            f(x) (tensor): batch x time x state dimension
          stable_type = f:
            f(x)=f'(x) - dv * scale
              scale = relu( dv f(x) )/||dv||^2
          stable_tye  = fgh:
            f(x)=f'(x) - dv * scale
              scale = relu( dv f(x) + k1^2 V_gh  )/||dv||^2
              V_gh = 1/(2gamma^2)|| gdv ||^2 + 1/2 d(h,h(D))


        """
        k2=1/2
        if self.stable_type=="f":
            x_=torch.tensor(x,requires_grad=True)
            v,i_v = self.func_v(x_)
            # v is independet w.r.t. batch and time
            dv = torch.autograd.grad(v.sum(), x_, create_graph=True)[0].detach()
            hj_dvf = torch.sum(dv * self.func_f(x_), dim=-1)
            dv_det = torch.sum(dv **2, dim=-1)+1.0e-10
            scale=F.relu(hj_dvf)/dv_det
            proj= dv * scale.unsqueeze(-1)
            f_new= self.func_f(x) - proj#.detach()
        elif self.stable_type=="fgh":
            hj_dvf, v_g, v_h, dv = self.compute_scale_fgh(x)
            v_gh = v_g+v_h
            dv_det = torch.sum(dv **2, dim=-1)+1.0e-10
            scale = F.relu(hj_dvf+k2*v_gh)/dv_det
            proj = dv * scale.unsqueeze(-1)
            f_new= self.func_f(x) - proj#.detach()
        elif self.stable_type=="fg":
            hj_dvf, v_g, v_h, dv = self.compute_scale_fgh(x)
            v_fh = hj_dvf+v_h
            dv_det = torch.sum(dv **2, dim=-1)+1.0e-10
            scale = F.relu(v_fh+k2*v_g)/dv_det
            proj = dv * scale.unsqueeze(-1)
            f_new= self.func_f(x) - proj#.detach()
        else:
            ## standard
            f_new = self.func_f(x)
        if self.max_value_clipping:
            f_new = torch.clip(f_new, -self.max_state_value, self.max_state_value)
        return f_new

    def compute_g(self,x):
        k2=1/2
        if self.stable_type=="fgh":
            g = self.func_g_mat(x)
            hj_dvf, v_g, v_h, dv = self.compute_scale_fgh(x)
            v_gh = v_g+v_h+1.0e-10
            #Pdv (tensor): batch (x time) x state dimension x state dimension 
            #g   (tensor): batch (x time) x input dimension x state dimension 
            pdv = self.compute_Pdv(dv)
            scale = torch.sqrt(torch.clip(-hj_dvf/v_gh,k2,1))
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            proj = (1-scale)*torch.matmul(g,pdv)
            #print("hj_dvf:",hj_dvf.shape)
            #print("v_gh:",v_gh.shape)
            #print("g:",g.shape)
            #print("pdv:",pdv.shape)
            #print("scale:",scale.shape)
            #print("proj:",proj.shape)
            g_new = g - proj#.detach()
        elif self.stable_type=="fg":
            g = self.func_g_mat(x)
            hj_dvf, v_g, v_h, dv = self.compute_scale_fgh(x)
            v_fh = hj_dvf+v_h
            #Pdv (tensor): batch (x time) x state dimension x state dimension 
            pdv = self.compute_Pdv(dv)
            scale = torch.sqrt(torch.clip(-v_fh/(v_g+1.0e-10),k2,1))
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            proj = (1-scale)*torch.matmul(g,pdv)
            g_new = g - proj#.detach()
        else:
            g_new = self.func_g_mat(x)
        if self.max_value_clipping:
            g_new = torch.clip(g_new, -self.max_state_value, self.max_state_value)
        return g_new

    def compute_h(self,x):
        k2=1/2
        if self.stable_type=="fgh":
            hj_dvf, v_g, v_h, dv = self.compute_scale_fgh(x)
            v_gh = v_g+v_h+1.0e-10
            scale = torch.sqrt(torch.clip(-hj_dvf/v_gh,k2,1))
            scale=scale.unsqueeze(-1)
            #print("hj_dvf:",hj_dvf.shape)
            #print("scale:",torch.mean(scale).item())
            #print("scale:",scale.shape)
            
            # h_star: batch (x time) x obs dimension
            _, h_star=self.compute_h_star(x)
            h_star= h_star.detach()
            h_hat = self.func_h(x)
            #print("h_star:",h_star.shape)
            #print("h_hat:",h_hat.shape)

            h_new = h_star+scale*(h_hat-h_star)
        else:
            h_new = self.func_h(x)
        if self.max_value_clipping:
            h_new = torch.clip(h_new, -self.max_obs_value, self.max_obs_value)
        return h_new
    
    def compute_h_star(self, x, i_v=None, hx=None):
        """
        Args:
            x (tensor):   batch x time x state dimension
            i_v (tensor): batch x time
        Returns:
            d(h(x),hmu)^2 (tensor): batch x time
            h_star (tensor): batch x time x observation dimension
        """
        if i_v is None:
            _,i_v = self.func_v(x)
        if hx is None:
            hx = self.func_h(x)
        ##
        mu_list=self.func_v.get_stable()
        hj_hh_list=[]
        hj_hh_pt_list=[]
        ## hh_pt: 
        for mu_type, mu in mu_list:
            if mu_type=="point":
                hh,hh_pt= self._distance(hx,mu, with_nearest_point=True)
            elif mu_type=="limit_cycle":
                hh,hh_pt= self._unit_limit_cycle_distance(hx, with_nearest_point=True)
            else:
                print("[ERROR] unknown:",mu_type)
            hj_hh_list.append(hh)
            hj_hh_pt_list.append(hh_pt)
        #
        if len(mu_list)==2:
            hj_hh=torch.where(i_v==1, hj_hh_list[1],hj_hh_list[0])
            h_star=torch.where(i_v==1, hj_hh_pt_list[1],hj_hh_pt_list[0])
            h_star=h_star.reshape(list(i_v.shape)+[self.obs_dim])
        elif len(mu_list)==1:
            hj_hh=hj_hh_list[0]
            h_star=hj_hh_pt_list[0]
            if mu_type=="limit_cycle":
                i_shape=[s for s in i_v.shape]+[self.obs_dim]
                o_shape=[1 for _ in i_v.shape]+[1]
            else:
                i_shape=[1 for _ in i_v.shape]+[self.obs_dim]
                o_shape=list(i_v.shape)+[1]
            h_star=h_star.reshape(i_shape)
            h_star=torch.tile(h_star,o_shape)
        elif len(mu_list)>2:
            hj_hh=hj_hh_list[0]
            h_star=hj_hh_pt_list[0]
            for i in range(len(mu_list)-1):
                hj_hh=torch.where(i_v==i+1, hj_hh_list[i+1],hj_hh)
                h_star=torch.where(i_v==i+1, hj_hh_pt_list[i+1],h_star)
            h_star=h_star.reshape(list(i_v.shape)+[self.obs_dim])
        ##
        return hj_hh, h_star



    def distance_h(self, x, i_v=None, hx=None):
        """
        Args:
            x (tensor):   batch x time x state dimension
            i_v (tensor): batch x time
        Returns:
            d(h(x),hmu)^2 (tensor): batch x time
        """
        if i_v is None:
            _,i_v = self.func_v(x)
        if hx is None:
            hx = self.func_h(x)
        # hh (tensor): batch x time
        # nearest_points (tensor): batch x time x state_dim
        mu_list=self.func_v.get_stable()
        hj_hh_list=[]
        for mu_type, mu in mu_list:
            if mu_type=="point":
                hh= self._distance(hx,mu)
            elif mu_type=="limit_cycle":
                hh= self._unit_limit_cycle_distance(hx)
            else:
                print("[ERROR] unknown:",mu_type)
            hj_hh_list.append(hh)
        #
        if len(mu_list)==2:
            hj_hh=torch.where(i_v==1, hj_hh_list[1],hj_hh_list[0])
        elif len(mu_list)==1:
            hj_hh=hj_hh_list[0]
        elif len(mu_list)>2:
            hj_hh=hj_hh_list[0]
            for i in range(len(mu_list)-1):
                hj_hh=torch.where(i_v==i+1, hj_hh_list[i+1],hj_hh)
        ##
        return hj_hh

    def compute_GdV(self, dv, gx):
        """
        Args:
            x (tensor): batch (x time) x state dimension
            g(x) (tensor): batch  (x time) x input dimension x state dimension
            dv (tensor): batch (x timev) x state dimension
        Returns:
            gdv2 (tensor): batch (x timev)  ||G dV||^2
        """
        #gx = self.func_g_mat(x)
        dv = torch.unsqueeze(dv, -1)
        #dv: batch (x time) x state dimension x 1
        #gx: batch (x time) x input dimension x state dimension 
        gdv = torch.matmul(gx, dv)
        gdv2 = torch.sum(gdv ** 2, dim=(-1, -2))
        return gdv2


    def compute_HJ(self, x):
        """
        x: batch x time x state dimension
        """
        ## dvf
        v,i_v = self.func_v(x)
        # v is independet w.r.t. batch and time
        dv = torch.autograd.grad(v.sum(), x, create_graph=True)[0].detach()
        hj_dvf = torch.sum(dv * self.compute_f(x), dim=-1)
        ## gamma
        if self.gamma is None:
            gamma = (1.0e-4+F.relu(self.param_gamma))
        else:
            gamma = self.gamma
        ## hh
        hj_hh = 1 / 2.0 * self.distance_h(x, i_v)
        ## gg
        gx = self.compute_g(x)
        gdv2 = self.compute_GdV(dv, gx)
        hj_gg = 1 / (2.0 * gamma ** 2 + 1.0e-10) * gdv2

        ## loss
        loss_hj = hj_dvf+ hj_hh + hj_gg
        if self.hj_loss_type=="const":
            loss_hj = F.relu(loss_hj + self.c)
        elif self.hj_loss_type=="cV":
            loss_hj = F.relu(loss_hj + self.c*v)
        return loss_hj, [hj_dvf, hj_hh, hj_gg]

    def vecmat_mul(self, vec, mat):
        """
        Args:
            vec: (...) x dim1
            mat: (...) x dim1 x dim2
        Returns:
            output: (...) x dim2
        """
        vec_m = torch.unsqueeze(vec, -2)
        o = torch.matmul(vec_m, mat)
        o = torch.squeeze(o, -2)
        return o

    def simulate_one_step(self, current_state, input_=None):
        if input_ is not None:
            ###
            g = self.compute_g(current_state)
            u = input_[:, :]
            ug = self.vecmat_mul(u, g)
            ###
            next_state = (
                current_state + (self.compute_f(current_state) + ug) * self.delta_t
            )
        else:
            next_state = current_state + self.compute_f(current_state) * self.delta_t
        return next_state

    def simulate(self, init_state, batch_size, step, input_=None):
        current_state = init_state
        state = []
        obs_generated = []
        for t in range(step):
            ## simulating system
            if input_ is None:
                next_state=self.simulate_one_step(current_state, None)
            else:
                next_state=self.simulate_one_step(current_state, input_[:, t, :])
            o = self.compute_h(current_state)
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
                init_state =torch.zeros((batch_size, self.state_dim),device=self.device)
                # mu: n_sample x 2
                init_state[:,:state.shape[2]] =state[:,0,:]
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
        state_generated, obs_generated = self.simulate(init_state, batch_size, step, input_)
        return state_generated, obs_generated

    def forward_state_sampling(self,n,m,scale):
        """
        Args:
            n (int): batch
            m (int): time
            scale (float): scale
        Returns:
            x (tensor): n*|mu_list| x m x state dimension
        """
        mu_list=self.func_v.get_stable()
        if len(mu_list)>0:
            ret=[]
            for mu_type,mu in mu_list:
                if mu_type=="point":
                    x= scale*torch.randn((n,m,self.state_dim),requires_grad=True,device=self.device)+mu
                elif mu_type=="limit_cycle":
                    x= scale*torch.randn((n,m,self.state_dim),requires_grad=True,device=self.device)
                else:
                    print("[ERROR] unknown:",mu_type)
                ret.append(x)
            ret=torch.cat(ret, dim=0)
            return ret
        return None

    def forward_loss(self, obs, input_, state, obs_generated, state_generated,
            with_state_loss=False,
            step_wise_loss=False,
            epoch=None):

        if epoch is not None and epoch < self.schedule_pretrain_epoch:
            self.stable_type="none"
            self.alpha={"recons":1.0,"HJ":0.0,"gamma":0.0,"state":0.0}
        else:
            self.stable_type=self.schedule_stable_type
            self.alpha=self.schedule_alpha
        ###
        ### observation loss
        ###
        #print("obs(r)",obs_generated.shape)
        #print("obs",obs.shape)
        # print("state",state.shape)
        loss_recons = (obs - obs_generated) ** 2
        
        ###
        ### HJ loss
        ###
        #state_rand =3*torch.randn(state.shape,requires_grad=True,device=self.device)
        state_rand =self.forward_state_sampling(1000,1,scale=3.0)
        loss_hj, loss_hj_list = self.compute_HJ(state_rand)
        #loss_hj, loss_hj_list = self.compute_HJ(state)
        ##
        if step_wise_loss:
            loss_sum_recons=loss_recons.sum(dim=2).mean(dim=(0,1))
            loss_sum_hj=loss_hj.mean(dim=(0,1))
        else:
            loss_sum_recons=loss_recons.sum(dim=(1,2)).mean(dim=0)
            loss_sum_hj=loss_hj.mean(dim=(0,1))
            #loss_sum_hj=loss_hj.sum(dim=1).mean(dim=0)
        ###
        ### summation loss & scheduling
        ###
        if epoch is not None and epoch<0:
            loss = {
                "recons": self.alpha["recons"]*loss_sum_recons,
                "*recons": loss_sum_recons,
            }
        else:
            loss = {
                "recons": self.alpha["recons"]*loss_sum_recons,
                "HJ": self.alpha["HJ"]*loss_sum_hj,
                "*recons": loss_sum_recons,
                "*HJ": loss_sum_hj,
                "*HJ_dvf": loss_hj_list[0].sum(dim=1).mean(dim=0),
                "*HJ_hh": loss_hj_list[1].sum(dim=1).mean(dim=0),
                "*HJ_gg": loss_hj_list[2].sum(dim=1).mean(dim=0),
            }
        
            ###
            ### IO stability loss
            ###
            if self.gamma is None:
                loss["gamma^2"]=self.alpha["gamma"]*self.param_gamma**2
                loss["*gamma^2"]=self.param_gamma**2
            
            ###
            ### state loss
            ###
            # state regularization
            """
            if state is not None:
                state_scale=1
                loss_state = (state) ** 2/(2*state_scale**2)
                loss_sum_state = loss_state.sum(dim=(1,2)).mean(dim=0)    
                loss["state-reg"] = self.alpha["state"] * loss_sum_state
                loss["*state-reg"] = loss_sum_state
            """
            # display state error
            if with_state_loss:
                if state is not None:
                    loss_state = (state - state_generated) ** 2
                    loss_sum_state = loss_state.sum(dim=(1,2)).mean(dim=0)
                    loss["*state-error"] = loss_sum_state

        
        return loss

    def forward(self, obs, input_, state=None, with_generated=False, epoch=None, step_wise_loss=True):
        state_generated, obs_generated = self.forward_simulate(obs, input_, state)
        if input_ is not None:
            l2_gain=self.forward_l2gain(obs, input_)
            l2_gain_recons=self.forward_l2gain(obs_generated, input_)
            loss=self.forward_loss(obs, input_, state_generated, obs_generated, state, epoch=epoch, step_wise_loss=step_wise_loss)
            loss["*l2"]=l2_gain.mean()
            loss["*l2_recons"]=l2_gain_recons.mean()
        else:
            loss=self.forward_loss(obs, input_, state_generated, obs_generated, state, epoch=epoch, step_wise_loss=step_wise_loss)
        if with_generated:
            hh=1 / 2.0 * self.distance_h(state_generated)
            return loss, state_generated, obs_generated, hh
        else:
            return loss


