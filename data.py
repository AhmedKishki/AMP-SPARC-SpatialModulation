from typing import Any
import torch
import numpy as np

from config import Config

class Data:
    def __init__(self, 
                 config : Config
                 ) -> None:
        self.B = config.B
        self.Lin = config.Lin
        self.Nt = config.Nt
        self.Na = config.Na
        self.device = config.device
        self.npdtype = config.npdatatype
        self.datatype = config.datatype
        self.symbols = config.symbols
        
    def generate(self) -> torch.Tensor:
        """_summary_

        Returns:
            torch.Tensor: size Batch, N_transmit_antenna*Lin, 1 
        """
        x = np.zeros((self.B, self.Nt, self.Lin), dtype=self.npdtype)
        for b in range(self.B):
            for l in range(self.Lin):
                Na = np.random.choice(self.Nt, size=self.Na, replace=False)
                x[b, Na, l] = np.random.choice(self.symbols, size=1)
        x = np.reshape(x, (self.B, -1, 1))
        return torch.tensor(x, device=self.device, dtype=self.datatype, requires_grad=False)
    
if __name__ == "__main__":
    config = Config(2, 6, 2, 5, 2, 5, 30, alphabet='8PSK')
    data = GSSK(config)
    print(config.symbols)
    print(data.generate().view(2, 6, -1))