import os

import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from bamp import BAMP


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.device = config.device
        self.sparsity = config.sparsity
        self.BAMP = BAMP(config).to(config.device)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = 'Simulations/BAMP,' + f'{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
    @torch.no_grad()
    def run(self, SNRdB: float) -> Loss:
        SNR = 10 ** ( SNRdB / 10)
        self.channel.generate(svd=False)
        x = self.data.generate()
        y = self.channel(x, SNR)
        loss = self.BAMP(x, y, self.channel)
        loss.save_stats(f'SNRdB={SNRdB}', self.path)
        return loss
    
    def simulate(self, start: float, final: float, step: float = 1):
        SNRdB_range = np.arange(start, final+step, step)
        for SNRdB in SNRdB_range:
            self.run(SNRdB)
        
if __name__ == "__main__":
    config = Config(batch=100,
                    N_transmit_antenna=200,
                    N_active_antenna=10,
                    N_receive_antenna=200,
                    block_length=20,
                    channel_length=10,
                    iterations=15,
                    alphabet='OOK')
    model = Model(config)
    model.simulate(0, 40, 1)