from typing import Any
import torch
import numpy as np

from config import Config

class Data:
    def __init__(self, 
                 config : Config
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
        self.symbols = config.symbols
        self.gray = config.gray
        self.info_bits = config.info_bits
        self.cardinality = len(self.symbols)
        
        if config.is_complex:
            self.dtype = torch.complex64
            self.npdtype = np.complex64
        else:
            self.dtype = torch.float32
            self.npdtype = np.float32
            
        if config.mode == 'random':
            self._generator = self.random
        elif config.mode == 'segmented':
            assert self.Nt % self.Na == 0,'Na must divide Nt'
            self.section = self.Nt // self.Na
            self._generator = self.segmented

    def generate_message(self) -> torch.Tensor:
        """_summary_

        Returns:
            torch.Tensor: size Batch, N_transmit_antenna*Lin, 1 
        """
        x, z, i = self._generator()
        x = torch.tensor(x, device=self.device, dtype=self.dtype, requires_grad=False)
        return x, z, i
    
    def random(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        x = np.zeros((self.B, self.Lin, self.Nt), dtype=self.npdtype)
        xgray = np.zeros((self.B, self.Lin, self.Nt), dtype=int) 
        for i in range(self.B):
            for j in range(self.Lin):
                space_index = np.random.choice(self.Nt, size=self.Na, replace=False)
                mod_index = np.random.choice(self.cardinality)
                x[i, j, space_index] = self.symbols[mod_index]
                xgray[i, j, space_index] = self.gray[mod_index]
        x = np.reshape(x, (self.B, -1, 1))
        index = x.ravel().nonzero()[0]
        symbol = xgray.ravel()[index]
        return x, symbol, index
    
    def segmented(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        x = np.zeros((self.B, self.Lin, self.Nt), dtype=self.npdtype)
        xgray = np.zeros((self.B, self.Lin, self.Nt), dtype=int) 
        for i in range(self.B):
            for j in range(self.Lin):
                space_index = [i*self.section + np.random.choice(self.section) for i in range(self.Na)]
                mod_index = np.random.choice(self.cardinality)
                x[i, j, space_index] = self.symbols[mod_index]
                xgray[i, j, space_index] = self.gray[mod_index]
        x = np.reshape(x, (self.B, -1, 1))
        index = x.ravel().nonzero()[0]
        symbol = xgray.ravel()[index]
        return x, symbol, index
    
if __name__ == "__main__":
    pass