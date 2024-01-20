from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

class Tracker:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, H: torch.Tensor, sigma2: float):
        self.y = y
        self.H = H
        self.sigma2 = sigma2
        self.adj = H.adjoint()
        self.abs2 = H.abs()**2
        self.abs2T = self.abs2.T
        self.xmmse = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.v = torch.zeros_like(y)
        self.z = y
        self.xmap = None
        self.u = self.v + self.sigma2
        
class BAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.Nt, self.Na, self.Lin, self.B = config.Nt, config.Na, config.Lin, config.B
        self.M = self.Nt // self.Na
        self.L = self.Na * self.Lin
        self.denoiser = self.segmented_shrinkage

    def forward(self,
                T: Tracker
                )-> None:
        """_summary_

        Args:
            y (torch.Tensor): _description_
            T (Tracker): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        T.v = T.abs2 @ T.var
        T.z = T.H @ T.xmmse - T.v * (T.y - T.z) / T.u
        T.u = T.v + T.sigma2
        cov = 1 / (T.abs2T @ (1 / T.u))
        T.xmap = T.xmmse + cov * (T.adj @ ((T.y - T.z) / T.u))
        T.xmmse, T.var = self.denoiser(T.xmap, cov)
        
    # def approx_shrinkage(self, r: torch.Tensor, cov: torch.Tensor):
    #     Lrx = ((2*r.real - 1) / cov).view(self.B, self.L, self.M)
    #     mask = ~torch.eye(self.M, dtype=torch.bool)
    #     ex_max = torch.zeros_like(Lrx)
    #     for i in range(self.M):
    #         ex_max[:, :, i], _ = Lrx[:, :, mask].max(dim=-1)
    #     eta1 = torch.exp(Lrx)
    #     eta2 = torch.exp(ex_max)
    #     exp = eta1 / (eta1 + eta2)
    #     var = eta1 * eta2 / (eta1 + eta2)**2
    #     return exp.view(-1, self.L*self.M, 1).to(torch.complex64), var.view(-1, self.L*self.M, 1).to(torch.float32)
    
    # def exact_shrinkage(self, r: torch.Tensor, cov: torch.Tensor):
    #     Lr = ((2*r.real - 1) / cov).view(self.B, self.L, self.M)
    #     mask = ~torch.eye(self.M, dtype=torch.bool)
    #     Le = torch.zeros_like(Lr)
    #     for i in range(self.M):
    #         Le[:, :, i] = torch.exp(Lr[:, :, mask[i]]).sum(dim=-1)
    #     Lx = (Lr - torch.log(Le)).view(-1, self.L*self.M, 1)
    #     eta1 = torch.exp(Lx)
    #     eta2 = torch.exp(1 + Lx)
    #     exp = eta1 / eta2
    #     var = exp / eta2
    #     return exp.to(torch.complex64), var.to(torch.float32)
    
    def segmented_shrinkage(self, r: torch.Tensor, cov: torch.Tensor):
        Lr = ((2*r.real - 1) / cov).view(self.B, self.L, self.M)
        Le = torch.zeros_like(Lr)
        exp_Lr = torch.exp(self.regularize(Lr))
        sum_exp_Lr = exp_Lr.sum(dim=-1)
        for i in range(self.M):
            Le[:, :, i] = - torch.log(sum_exp_Lr - exp_Lr[:, :, i])
        Lx = Lr + Le
        eta = torch.exp(self.regularize(Lx))
        Exp = eta / (1 + eta)
        Var = Exp * (1 - Exp)
        return Exp.to(torch.complex64).view(self.B, self.L*self.M, 1), Var.to(torch.float32).view(self.B, self.L*self.M, 1)
    
    def regularize(self, a: torch.Tensor):
        max = np.log(torch.finfo(a.dtype).max)
        a[a>=max] = max - 1
        return a
    
class BAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        
        self.layers = nn.ModuleList([BAMPLayer(config) for _ in range(config.N_Layers)])
        self.L = Loss(config)

    def forward(self,
                H: torch.Tensor,
                y: torch.Tensor,
                SNR: float,
                x: torch.Tensor, 
                symbols,
                indices
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            xmmse: _description_
        """
        T = Tracker(x, y, H, self.E / SNR)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            var_prev = T.var
            layer(T)
            var_next = T.var
            if torch.allclose(var_next, var_prev):
                break
        np.savetxt('xmap.txt', T.xmap.view(-1).cpu().numpy())
        self.L(T.xmap, T.xmmse, x, symbols, indices, t+1)
        return self.L
    
        
