from typing import Tuple
import torch
from torch import nn

from config import Config

class Shrink(nn.Module):
    def __init__(self, config: Config, shrink_fn: str):
        """_summary_

        Args:
            config (Config): _description_
            shrink_fn (str): _description_
        """
        super().__init__()
        assert shrink_fn in ["bayes", "shrink", "lasso"], "shrink_fn needs to be 'bayes' or 'shrink' or 'lasso'"
         
        self.Ps, self.P0 = config.Ps, config.P0
        self.datatype = config.datatype 
        self.lmda = config.lmda
        self.symbols = torch.tensor(config.symbols, device=config.device)
        self.symbols2 = torch.abs(self.symbols)**2
        self.tol = 1.1e-20
        
        if shrink_fn == "bayes":
            self.shrinkage = self.bayes
        elif shrink_fn == "shrink":
            self.shrinkage = self.shrink
        elif shrink_fn == "lasso":
            self.shrinkage = self.lasso
            
    def forward(self, 
                  r: torch.Tensor, 
                  cov: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            cov (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        return self.shrinkage(r, cov)
    
    def bayes(self, 
                  r: torch.Tensor, 
                  cov: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            u (torch.Tensor): _description_
            v (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        G = lambda s: torch.exp(- torch.abs(r - s)**2 / cov )
        G0, Gs = G(0), G(self.symbols)
        norm = self.P0 * G0 + self.Ps * torch.sum(Gs, dim=-1).unsqueeze(-1) + self.tol
        exp = self.Ps * torch.sum(self.symbols * Gs, dim=-1).unsqueeze(-1) / norm
        var = self.Ps * torch.sum(self.symbols2 * Gs, dim=-1).unsqueeze(-1) / norm - torch.abs(exp)**2
        return exp, var.to(torch.float32)
    
    def shrink(self, 
                r: torch.Tensor, 
                cov: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            cov (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        G = lambda s: torch.exp(- torch.abs(r - s)**2 / cov )
        d = lambda s: 2 * torch.abs(r - s) * torch.sign(s - r) / cov
        G0, Gs, d0, ds, dG0, dGs = G(0), G(self.symbols), d(0), d(self.symbols), d0 * G0, ds * Gs
        norm = self.P0 * G0 + self.Ps * torch.sum(Gs, dim=-1).unsqueeze(-1) + self.tol
        exp = self.Ps * torch.sum(self.symbols * Gs, dim=-1).unsqueeze(-1) / norm
        dG = self.P0 * dG0 + self.Ps * torch.sum(dGs, dim=-1).unsqueeze(-1)
        sdGs = self.Ps * torch.sum(self.symbols * dGs, dim=-1).unsqueeze(-1)
        der = ( sdGs * norm - exp * dG ) / norm**2
        return exp, der
    
    def lasso(self, 
                r: torch.Tensor, 
                cov: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            cov (torch.Tensor): _description_
            lmda (float): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        F = torch.sign(r) * torch.max(torch.abs(r), self.lmda * cov)
        G = cov * torch.where(torch.abs(r) < cov, 0.0, 1.0)
        return F, G
        
    def shrinkOOK(self, 
                  r: torch.Tensor, 
                  cov: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            cov (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        theta = torch.log(self.P0 / self.Ps)
        eta = torch.exp(theta + (1 - 2*r) / cov)
        exp = 1 / (1 + eta + self.tol)
        der = 2 * eta * exp**2 / (cov + self.tol)
        return exp, der