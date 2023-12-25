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
        self.M = self.Nt * self.Lin
        self.Ns, self.sparsity = config.Ns, config.sparsity
        # self.info_bits = config.info_bits
        self.symbols = torch.tensor(config.symbols, dtype=config.datatype)
        # self.gray = torch.tensor(config.gray)
        self.datatype = config.datatype
        self.device = config.device

    def __call__(self, 
                 xamp: torch.Tensor, 
                 x: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            x (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        xamp = xamp.cpu()
        x = x.cpu()
        xhat = self.xhat(xamp)
        MSE = torch.sum(torch.abs(xamp - x)**2) / self.Ns
        VER = torch.count_nonzero(xhat - x) / self.Ns
        SER = (torch.sum(xhat.view(-1, self.Nt) - x.view(-1, self.Nt), dim=-1) != 0).sum() / self.Lin / self.B
        FER = (torch.sum(xhat.view(-1, self.M) - x.view(-1, self.M), dim=-1) != 0).sum() / self.B
        self.MSE = np.append(self.MSE, MSE.item()) # Mean square error
        self.VER = np.append(self.VER, VER.item()) # maximum number of errors is 2 times the number of ones
        self.SER = np.append(self.SER, SER.item()) # Section error rate
        self.FER = np.append(self.FER, FER.item()) # Frame error rate
        
    def export(self, name: str, save_location: str) -> None:
        """_summary_

        Args:
            name (str): _description_
        """
        stats = np.zeros((4, len(self.SER)))
        stats[0] = self.MSE
        stats[1] = self.VER
        stats[2] = self.SER
        stats[3] = self.FER
        np.savetxt(f'{save_location}/{name}.csv', stats, delimiter=",")

    def init(self):
        """
        initialise between epochs
        """
        # # initial values
        self.MSE = np.array([])
        self.VER = np.array([])
        self.SER = np.array([])
        self.FER = np.array([])
    
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
        xamp = xamp.view(-1)
        args = torch.topk(torch.abs(xamp), k=self.Ns).indices
        xhat = torch.zeros_like(xamp)
        for j, xs in enumerate(xamp[args]):
            d = torch.inf
            for s in self.symbols:
                ds = torch.abs(xs - s)**2
                if ds < d:
                    d = ds
                    xhat[args[j]] = s
        return xhat.view(self.B, -1, 1)
    
    # def measure(self, xamp: torch.Tensor, x:torch.Tensor, z: torch.Tensor):
    #     """_summary_

    #     Args:
    #         xamp (torch.Tensor): _description_
    #         x (torch.Tensor): _description_
    #         z (torch.Tensor): _description_
    #     """
    #     x, z, xamp = x.cpu().view(-1, self.Nt), z.cpu().view(-1), xamp.cpu().view(-1, self.Nt)
    #     xhat, zhat = self.xhat_section(xamp)
    #     SpatialBitER = torch.count_nonzero(torch.abs(xhat) - torch.abs(x)).item()
    #     ModulationBitER = sum(bin(a ^ b).count('1') for a, b in zip(zhat, z))
    #     BER = (SpatialBitER + ModulationBitER) / self.info_bits
    #     SER = (torch.sum(xhat - x, dim=-1) != 0).sum().item() / self.Lin / self.B
    #     FER = (torch.sum(xhat.view(-1, self.M) - x.view(-1, self.M), dim=-1) != 0).sum().item() / self.B
    #     MSE = torch.sum(torch.abs(xamp - x)**2).item() / self.Ns
    #     return BER, SER, FER, MSE
        
    # def xhat_section(self, xamp: torch.Tensor) -> torch.Tensor:
    #     """_summary_

    #     Args:
    #         xamp (torch.Tensor): _description_

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     symbol, index = self.sectiontopNa(xamp)
    #     xhat = torch.zeros(self.B, self.Lin, self.Nt, dtype=self.datatype)
    #     zhat = torch.zeros(self.B, self.Lin, dtype=torch.uint8)
    #     for i in range(self.B):
    #         for j in range(self.Lin):
    #             xs = symbol[i, j]
    #             d = torch.inf
    #             for k, s in zip(self.gray, self.symbols):
    #                 ds = torch.abs(xs - s)**2
    #                 if ds < d:
    #                     d = ds
    #                     xhat[i, j, index[i, j].int()] = s
    #                     zhat[i, j] = k
    #     xhat = xhat.view(-1, self.Nt)
    #     zhat = zhat.view(-1)
    #     return xhat, zhat
        
    # def sectiontopNa(self, xamp: torch.Tensor) -> torch.Tensor:
    #     """_summary_

    #     Args:
    #         xamp (torch.Tensor): _description_

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     xamp = xamp.view(self.B, self.Lin, self.Nt)
    #     symbol = torch.zeros(self.B, self.Lin, dtype=self.datatype)
    #     index = torch.zeros(self.B, self.Lin, self.Na)
    #     for i in range(self.B):
    #         for j in range(self.Lin):
    #             values, indices = torch.abs(xamp[i, j]).topk(k=self.Na)
    #             best = values.topk(k=1).indices
    #             symbol[i, j] =  xamp[i, j, indices][best]
    #             index[i, j] = indices
    #     return symbol, index
  
    def accumulate(self, other):
        """_summary_

        Args:
            other (_type_): _description_
        """
        if len(self.SER) == len(other.SER):
            self.SER += other.SER
            self.VER += other.VER
            self.MSE += other.MSE 
            self.FER += other.FER
        elif len(self.SER) == 0:
            self.SER = other.SER
            self.VER = other.VER
            self.MSE = other.MSE 
            self.FER = other.FER
        else:
            print(len(self.SER), len(other.SER), self.SER, other.SER)
            raise Exception("something bad happen")
            
    def average(self, epochs: int):
        self.SER = self.SER / epochs
        self.VER = self.VER / epochs
        self.MSE = self.MSE / epochs
        self.FER = self.FER / epochs