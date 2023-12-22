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

    @torch.no_grad()
    def run(self, SNR_db: float) -> Loss:
        SNR= 10 ** ( SNR_db / 10)
        print(SNR)
        self.channel.generate(svd=False)
        x = self.data.generate()
        y = self.channel(x, SNR)
        loss = self.BAMP(x, y, self.channel)
        return loss
    
    def simulate(self, runs, SNR_db_start, SNR_db_final, SNR_db_step):
        pass
        
if __name__ == "__main__":
    config = Config(batch=50, 
                    N_transmit_antenna=200,
                    N_active_antenna=10,
                    N_receive_antenna=20,
                    block_length=20,
                    channel_length=10,
                    iterations=15,
                    alphabet='OOK')
    model = Model(config)
    loss = model.run(15.)
    print(loss.SER)
    print(loss.VER)
    print(loss.hardMSE)
    print(loss.softMSE)