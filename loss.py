from typing import Tuple
import torch
import numpy as np

from config import Config

class Loss:
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.B, self.Nt, self.Na, self.Lin = config.B, config.Nt, config.Na, config.Lin
        self.Ns, self.sparsity = config.Ns, config.sparsity
        self.symbols = torch.tensor(config.symbols)
        self.datatype = config.datatype
        self.device = config.device

    def __call__(self, 
                 xamp:torch.Tensor, 
                 x:torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            x (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        x, xamp, xhat = x.cpu(), xamp.cpu(), self.xhat(xamp)
        Xhat, X = xhat.view(-1, self.Lin, self.Nt), x.view(-1, self.Lin, self.Nt)
        MSE = torch.mean(torch.abs(xhat - x)**2) / self.sparsity / 2
        BER = torch.count_nonzero(xhat - x) / self.Ns / 2
        SER = (torch.sum(Xhat - X, dim=-1) > 0).sum() / self.Lin
        self.MSE.append(MSE.item())
        self.BER.append(BER.item()) # maximum number of errors is 2 times the number of ones
        self.SER.append(SER.item()) #  Section error rate
        
        return MSE, BER, SER
        
    def save_stats(self, name: str, save_location: str) -> None:
        """_summary_

        Args:
            name (str): _description_
        """
        stats = np.zeros((3, len(self.SER)))
        stats[0] = self.MSE
        stats[2] = self.BER
        stats[3] = self.SER
        np.savetxt(f'{save_location}/{name}.csv', stats, delimiter=",")
        
    def init(self):
        """
        initialise between epochs
        """
        # initial values
        self.MSE = [1.]
        self.SER = [1.0]
        self.BER = [0.5]
    
    def xhat(self, xamp: torch.Tensor) -> torch.Tensor:
        """
        TopK decision: Largest B*Na*Lin terms are assumed to be nonzero, while the rest are assumed to be zero.
        Hard decision is then performed on the non zero set.
        
        !!!Does not perserve the original structure of x, and may have more than Na active antennas per section!!!

        Args:
            xamp (torch.Tensor): _description_

        Returns:
            torch.Tensor: cpu
        """
        xamp = xamp.cpu().view(-1)
        args = torch.topk(torch.abs(xamp), k=self.Ns).indices
        xhat = torch.zeros_like(xamp)
        
        for i, xs in enumerate(xamp[args]):
            d = torch.inf
            for s in self.symbols:
                ds = torch.abs(xs - s)**2
                if ds < d:
                    d = ds
                    xhat[args[i]] = s
                    
        return xhat.view(self.B, -1, 1).to(self.device)
    
    def demodulation(self, xamp: torch.Tensor, mapping='gray'):
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            mapping (str, optional): _description_. Defaults to 'gray'.
        """
        pass