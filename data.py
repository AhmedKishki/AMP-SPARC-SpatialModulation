from typing import Any
import torch
import numpy as np

from config import Config

class Data:
    def __init__(self, 
                 config : Config,
                 mode: str = 'random'
                 ) -> None:
        """_summary_

        Args:
            config (Config): _description_
            mode (str, optional): _description_. Defaults to 'random'.

        Raises:
            ValueError: _description_
        """
        self.B = config.B
        self.Lin = config.Lin
        self.Nt = config.Nt
        self.Na = config.Na
        self.Ns = config.Na
        self.device = config.device
        self.npdtype = config.npdatatype
        self.datatype = config.datatype
        self.symbols = config.symbols
        self.cardinality = len(self.symbols)
        
        if mode == 'random':
            self._generator = self.random
        elif mode == 'sparcs':
            if self.Nt % self.Na != 0:
                raise ValueError('Na must divide Nt')
            else:
                self._generator = self.sparcs
        else:
            raise Exception('something bad happen')
        
    def generate(self) -> torch.Tensor:
        """_summary_

        Returns:
            torch.Tensor: size Batch, N_transmit_antenna*Lin, 1 
        """
        x, z, i = self._generator()
        x = torch.tensor(x, device=self.device, dtype=self.datatype, requires_grad=False)
        return x, z, i
    
    def random(self):
        x = np.zeros((self.B, self.Lin, self.Nt), dtype=self.npdtype)
        indices = np.zeros((self.B, self.Lin, self.Na))
        symbols = np.zeros((self.B, self.Lin, 1))              
        for i in range(self.B):
            for j in range(self.Lin):
                mod_index = np.random.choice(self.cardinality)
                space_index = np.random.choice(self.Nt, size=self.Na, replace=False)
                symbols[i, j] = mod_index
                indices[i, j] = space_index
                x[i, j, space_index] = self.symbols[mod_index]
        x = np.reshape(x, (self.B, -1, 1))
        symbols = np.reshape(symbols, -1)
        indices = np.reshape(indices, -1)
        return x, symbols, indices
    
    def sparcs(self):
        x = np.zeros((self.B, self.Lin, self.Nt), dtype=self.npdtype)
        indices = np.zeros((self.B, self.Lin, self.Na))
        symbols = np.zeros((self.B, self.Lin, 1))          
        for i in range(self.B):
            for j in range(self.Lin):
                mod_index = np.random.choice(self.cardinality)
                space_index = [(self.Na)*i + np.random.choice(self.Nt // self.Na) for i in range(self.Na)]
                symbols[i, j] = mod_index
                indices[i, j] = space_index
                x[i, j, space_index] = self.symbols[mod_index]
        x = np.reshape(x, (self.B, -1, 1))
        symbols = np.reshape(symbols, -1)
        indices = np.reshape(indices, -1)
        return x, symbols, indices
        
    
if __name__ == "__main__":
    pass