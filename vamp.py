from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

class Tracker:
    def __init__(self, U:torch.Tensor, s:torch.Tensor, Vh:torch.Tensor, y:torch.Tensor, x:torch.Tensor, sigma2:float, sparsity: float):
        self.U = U
        self.Uh = U.adjoint()
        self.s = s
        self.s2 = s**2
        self.Vh = Vh
        self.V = Vh.adjoint()
        self.noise_var = sigma2
        self.sigma2 = torch.tensor(sigma2)
        self.y_tilde = s.view(-1, 1) * self.Uh @ y
        self.r = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.r_tilde = torch.ones_like(x) * sparsity
        self.sigma2_tilde = (sparsity)**2 * (1-sparsity) + (1-sparsity)**2 * sparsity
        self.xmmse = None

class VAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
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
        
        self.var_ratio_min = torch.tensor(1.0e-5)
        self.var_ratio_max = 1.0 - self.var_ratio_min
        self.var_min = torch.tensor(1.0e-9)
        self.var_max = torch.tensor(1.0e5)
        
    def forward(self,
                T: Tracker):
        """_summary_

        Args:
            T (Tracker): _description_

        Returns:
            _type_: _description_
        """
        var_ratio = T.noise_var / T.sigma2_tilde
        q = T.Vh @ T.r_tilde
        scale = 1 / (T.s2 + var_ratio)
        
        x_tilde = scale.view(-1, 1) * (T.y_tilde + var_ratio * q)
        varLMMSE = scale.mean() * T.noise_var
        x_tilde = T.V @ (x_tilde - q) + T.r_tilde
        x_tilde_var = self.Nr/self.Nt * varLMMSE + (1 - self.Nr/self.Nt) * T.sigma2_tilde

        alpha = x_tilde_var / T.sigma2_tilde
        alpha = torch.max(alpha, self.var_ratio_min)
        alpha = torch.min(alpha, self.var_ratio_max)
        
        T.r = (x_tilde - alpha * T.r_tilde) / (1 - alpha)
        sigma2 = alpha/(1-alpha)*T.sigma2_tilde
        sigma2 = torch.max(sigma2, self.var_min) 
        sigma2 = torch.min(sigma2, self.var_max)
        
        T.xmmse, T.var = self.denoiser(T.r, sigma2)
        dxdr = T.var.mean() / sigma2
        dxdr = torch.max(dxdr, self.var_ratio_min)
        dxdr = torch.min(dxdr, self.var_ratio_max)
        
        normScalar = 1 / (1 - dxdr)
        
        T.r_tilde = (T.xmmse - dxdr * T.r) * normScalar
        T.sigma2_tilde = sigma2 * dxdr * normScalar
        T.sigma2_tilde = torch.max(T.sigma2_tilde, self.var_min)
        T.sigma2_tilde = torch.min(T.sigma2_tilde, self.var_max)
        
    def segmented_shrinkage(self, r: torch.Tensor, cov: torch.Tensor):
        Lr = ((2*r.real - 1) / cov).view(self.B, self.L, self.M)
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
        T = Tracker(U, s, Vh, y, x, self.E / SNR, self.sparsity)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            prev = T.var
            layer(T)
            next = T.var
            if torch.allclose(next, prev):
                break
        self.L(T.xmmse, T.xmmse, x, symbols, indices, t+1)
        # np.savetxt('r.txt', T.r.view(-1).cpu().numpy())
        # np.savetxt('xmmse.txt', T.xmmse.view(-1).cpu().numpy())
        # np.savetxt('var.txt', T.var.view(-1).cpu().numpy())
        return self.L