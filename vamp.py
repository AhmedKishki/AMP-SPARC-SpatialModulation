from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from channel import Channel


class VAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        VAMP iteration (layer)

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.Nr, self.Nt = config.Nr, config.Nt
        self.P0, self.P1 = config.P0, config.P1
        self.datatype = config.datatype
        self.factor = 2
        if config.is_complex:
            self.factor = 1
        
        # constraints for numerical stability
        self.var_min = torch.tensor(1.0e-5, requires_grad=False)
        self.var_max = 1.0 - self.var_min
        self.sigma2t_min = torch.tensor(1.0e-9, requires_grad=False)
        self.sigma2t_max = torch.tensor(1.0e5, requires_grad=False)
        
    def forward(self, 
                yt: torch.Tensor, 
                rt: torch.Tensor, 
                sigma2t: torch.Tensor,
                theta: torch.Tensor,
                channel: Channel,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            yt (torch.Tensor): _description_
            rt (torch.Tensor): _description_
            sigma2t (torch.Tensor): _description_
            theta (torch.Tensor): tied learnable shrinkage parameter
            sigma2_N (float): noise variance
            channel (Channel): channel instance: includes the svd

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        # var_ratio = channel.sigma2_N / sigma2t
    
        # "BSR17":
        # # Linear MMSE
        # d = 1/(channel.s2 + var_ratio)
        # zt = d * (channel.s2 * yt + channel.Vh @ (var_ratio * rt))
        # beta = var_ratio * d.mean(dim=-1).unsqueeze(-1)
        # xt = channel.V @ (zt - channel.Vh @ rt) + rt
        
        # # Onsager Correction
        # corr = 1 + self.Nr/self.Nt * (beta - 1)
        # # corr = torch.max(corr, self.var_min)
        # # corr = torch.min(corr, self.var_max)
        # r = (xt - corr * rt) / (1 - corr)
        # sigma2 = corr / (1 - corr) * sigma2t
        
        # # Shrinkage (non linear MMSE)
        # xhat, dxhat_dr = self.shrinkage(r, sigma2, theta)
        # alpha = dxhat_dr.mean(dim=1).unsqueeze(-1)
        # print(alpha.size(), xhat.size())
        # # alpha = torch.max(alpha, self.var_min)
        # # alpha = torch.min(alpha, self.var_max)
        # rt_next = (xhat - alpha * r) / (1 - alpha)
        # sigma2t_next = alpha / (1 - alpha) * sigma2
        # # sigma2t_next = torch.max(sigma2t_next, self.sigma2t_min)
        # # sigma2t_next = torch.min(sigma2t_next, self.sigma2t_max)
        
        # # return xhat, rt_next, sigma2t_next
    
        # "RSAK17":
        # d = 1 / (channel.s2 + var_ratio) * channel.s2
        # D = torch.diag_embed(d.squeeze(1))
        # xt = channel.V @ D @ (yt - channel.Vh @ rt)
        # alpha = d.mean(dim=-1).unsqueeze(-1)
        # r = rt + self.Nt / R * xt / alpha
        # sigma2 = sigma2t * (self.Nt/R - alpha) / alpha
        
        # xhat, dxhat_dr = self.shrinkage(r, sigma2, theta)
        # alpha = dxhat_dr.mean(dim=1).unsqueeze(-1)
        # rt_next = (xhat - alpha * r) / (1 - alpha)
        # sigma2t_next = alpha / (1 - alpha) * sigma2
        
        #"Rangan":
        # d = channel.s2 / (channel.s2 + var_ratio)
        # q = channel.Vh @ rt
        # alpha = d.mean(dim=-1).unsqueeze(-1)
        # xt = (channel.V * alpha)  @ (yt - q) / alpha
        # r = rt + self.Nt/self.Nr * xt
        # sigma2 = sigma2t * (self.Nt/self.Nr - alpha) / alpha
        # xhat, dxhatdr = self.shrinkage(r, sigma2, theta)
        # beta = dxhatdr.mean(dim=1).unsqueeze(-1)
        # rt_next = (xhat - beta * r)/(1 - beta)
        # sigma2t_next = sigma2 * beta/(1 - beta)

        #"vampyre":
        

        return xhat, rt_next, sigma2t_next

    def shrinkage(self, 
                  r: torch.Tensor, 
                  sigma2: torch.Tensor,
                  theta: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            sigma2 (torch.Tensor): _description_
            theta (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        r_ = r.real
        expo = torch.exp(theta + (1. - 2*r_) / sigma2 / self.factor)
        xhat = 1. / (1. + expo)
        dxhatdr = 2 * expo * xhat**2 / sigma2 / self.factor
        dxhatdr = torch.nan_to_num(dxhatdr)
    
        return xhat.to(self.datatype), dxhatdr.to(self.datatype)

class VAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        VAMP model

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.device = config.device
        self.xsize = config.insize
        self.B = config.B
        self.sparsity = config.sparsity
        self.datatype = config.datatype
        self.N_Layers = config.N_Layers
        self.sigma2t_init = self.sparsity**2 * (1-self.sparsity) + (1-self.sparsity)**2 * self.sparsity
        
        # setup
        self.layers = nn.ModuleList([VAMPLayer(config) for _ in range(self.N_Layers)])
        self.MSE = lambda xhat, x: 10*torch.log10(nn.MSELoss()(xhat, x))
        self.ERate = lambda xhat, x: 10*torch.log10(torch.count_nonzero(torch.round(xhat)  - x))
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor, 
                channel: Channel,
                ) -> torch.Tensor:
        """_summary_

        Args:
            y (torch.Tensor): _description_
            sigma2_N (float): _description_
            channel (Channel): _description_

        Returns:
            torch.Tensor: _description_
        """
        yt = torch.diag(1/channel.s) @ channel.Uh @ y
        rt = torch.ones(self.xsize, dtype=self.datatype, requires_grad=False, device=self.device) * self.sparsity
        sigma2t = torch.ones(self.B, 1, 1, requires_grad=False, device=self.device) * self.sigma2t_init
        mse = []
        error = []
        for i, layer in enumerate(self.layers):
            xhat, rt, sigma2t = layer(yt, rt, sigma2t, self.theta, channel)
            mse.append(self.MSE(xhat, x).item())
            error.append(self.ERate(xhat, x).item())
        return xhat, mse, error
         
if __name__ == "__main__":
    pass