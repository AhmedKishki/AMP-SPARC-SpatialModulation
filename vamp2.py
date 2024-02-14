from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

class Tracker:
    def __init__(self, U:torch.Tensor, s:torch.Tensor, Vh:torch.Tensor, y:torch.Tensor, x:torch.Tensor, sigma2:float):
        self.U = U
        self.Uh = U.adjoint()
        self.s = s.view(-1, 1)
        self.s2 = self.s**2
        self.Vh = Vh
        self.V = Vh.adjoint()
        self.sigma2 = sigma2
        self.gamma = torch.tensor(1.0)
        self.y_tilde = (self.Uh @ y) / self.s
        self.r = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.xmmse = torch.zeros_like(x)
        self.eta = Vh.size(1) / s.size(0)

class VAMPLayer(nn.Module):
    def __init__(self, config: Config, damping=0.97) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.Nt, self.Na, self.Nr, self.Lin, self.B = config.Nt, config.Na, config.Nr, config.Lin, config.B
        self.M = self.Nt // self.Na
        self.L = self.Na * self.Lin
        # self.denoiser = Shrink(config, 'shrinkOOK')
        self.denoiser = self.segmented_shrinkage
        
        self.var_min = torch.tensor(1.0e-11)
        self.var_max = torch.tensor(1.0e11)
        self.rho = damping
        
    def forward(self,
                T: Tracker):
        """Direct implementation of Rangan (with damping)

        Args:
            T (Tracker): _description_

        Returns:
            _type_: _description_
        """
        xmmse, T.var = self.denoiser(T.r, T.gamma)
        T.xmmse = self.rho * xmmse + (1 - self.rho) * T.xmmse
        alpha = T.var.mean() * T.gamma
        
        r_tilde = (T.xmmse - alpha * T.r) / (1 - alpha)
        gamma_tilde = T.gamma * (1 - alpha) / alpha
        gamma_tilde = torch.max(gamma_tilde, self.var_min)
        gamma_tilde = torch.min(gamma_tilde, self.var_max)
        
        d = T.s2 / (T.s2 + T.sigma2 * gamma_tilde) 
        gamma = gamma_tilde * d.mean() / (T.eta - d.mean())
        T.gamma = self.rho * gamma + (1 - self.rho) * T.gamma
        gamma = torch.max(gamma_tilde, self.var_min)
        gamma = torch.min(gamma_tilde, self.var_max)
        
        T.r = r_tilde + T.eta * T.V @ ((d / d.mean()) * (T.y_tilde - T.Vh @ r_tilde))
        
    def segmented_shrinkage(self, r: torch.Tensor, gamma: torch.Tensor):
        Lr = ((2*r.real - 1)*gamma).view(self.B, self.L, self.M)
        exp_Lr = torch.exp(self.regularize(Lr))
        sum_exp_Lr = exp_Lr.sum(dim=-1, keepdim=True).repeat_interleave(self.M, dim=-1)
        Le = - torch.log(sum_exp_Lr - exp_Lr)
        Lx = Lr + Le
        eta = torch.exp(self.regularize(Lx))
        Exp = eta / (1 + eta)
        Var = Exp * (1 - Exp)
        return Exp.to(torch.complex64).view(self.B, self.L*self.M, 1), Var.to(torch.float32).view(self.B, self.L*self.M, 1)
    
    def regularize(self, a: torch.Tensor):
        max = np.log(torch.finfo(a.dtype).max)
        a[a>=max] = max - 1
        return a
        
class VAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        self.sparsity = config.Na / config.Nt
        self.layers = nn.ModuleList([VAMPLayer(config) for _ in range(config.N_Layers)])
        self.L = Loss(config)

    def forward(self,
                U: torch.Tensor,
                s: torch.Tensor,
                Vh: torch.Tensor,
                y: torch.Tensor,
                SNR: float,
                x: torch.Tensor, 
                symbols: np.ndarray,
                indices: np.ndarray
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            xmmse: _description_
        """
        T = Tracker(U, s, Vh, y, x, self.E / SNR)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            prev = T.var
            layer(T)
            next = T.var
            if torch.allclose(next, prev):
                break
        self.L(T.r, T.xmmse, x, symbols, indices, t+1)
        # np.savetxt('r.txt', T.r.view(-1).cpu().numpy())
        # np.savetxt('xmmse.txt', T.xmmse.view(-1).cpu().numpy())
        # np.savetxt('var.txt', T.var.view(-1).cpu().numpy())
        return self.L