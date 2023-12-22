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

    def __call__(self, xamp:torch.Tensor, x:torch.Tensor) -> None:
        """
        call per epoch

        Args:
            xamp (torch.Tensor): _description_
            x (torch.Tensor): _description_
        """
        x, xamp, xhat = x.cpu(), xamp.cpu(), self.xhat(xamp)
        self.softMSE.append((torch.mean(torch.abs(xamp - x)**2) / self.sparsity / 2).item())
        self.hardMSE.append((torch.mean(torch.abs(xhat - x)**2) / self.sparsity / 2).item())
        self.ErrorRate(xhat, x)
        self.Statistics(xhat, x)
        
    def init(self):
        """
        initialise between epochs
        """
        # initial values
        self.hardMSE = [1.]
        self.softMSE = [1.]
        self.SER = [0.5]
        self.VER = [0.5]
        # self.AER = [0.5]
        self.ErMean = [torch.mean(self.symbols).item() * self.sparsity]
        self.ErVar = [torch.mean((torch.abs(self.symbols)**2)).item() * self.sparsity - abs(self.ErMean[0])**2]

    def MeanSqError(self, xhat: torch.Tensor, x:torch.Tensor) -> None:
        """_summary_

        Args:
            xhat (torch.Tensor): _description_
            x (torch.Tensor): _description_
        """
        NMSE = torch.mean(torch.abs(xhat - x)**2) / torch.mean(torch.abs(x)**2 / 2)
        self.softMSE.append(NMSE.item())

    def ErrorRate(self, xhat: torch.Tensor, x:torch.Tensor) -> None:
        """_summary_

        Args:
            xhat (torch.Tensor): _description_
            x (torch.Tensor): _description_
        """
        Xhat, X = xhat.view(-1, self.Nt, self.Lin), x.view(-1, self.Nt, self.Lin)
        SER = torch.count_nonzero(xhat - x) / self.Ns / 2 # maximum number of errors is 2 times the number of ones
        VER = torch.mean(torch.count_nonzero(Xhat - X, dim=1).to(float)) / self.Na / 2 #  Vector error rate
        # AER = torch.mean(torch.count_nonzero(Xhat - X, dim=2).to(float)) / self.Na / 2 # Antenna error rate
        self.SER.append(SER.item())
        self.VER.append(VER.item())
        # self.AER.append(AER.item())
    
    def xhat(self, xamp: torch.Tensor) -> torch.Tensor:
        """
        TopK decision: Largest B*Na*Lin terms are assumed to be nonzero, while the rest are assumed to be zero.
        Hard decision is then performed on the non zero set.
        
        !!!Does not perserve the original structure of x, and may have more than Na active antennas per symbol!!!

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
                    
        return xhat.view(self.B, -1, 1)
    
    def demodulation(self, xamp: torch.Tensor, mapping='gray'):
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            mapping (str, optional): _description_. Defaults to 'gray'.
        """
        xhat = self.xhat(xamp).view(-1, self.Nt, self.Lin)
        pass
        
    def Statistics(self, xhat: torch.Tensor, x:torch.Tensor) -> None:
        """_summary_

        Args:
            xhat (torch.Tensor): _description_
            x (torch.Tensor): _description_
        """
        self.ErMean = torch.mean(xhat - x)
        self.ErVar = torch.std(xhat - x)**2