import os

import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from vamp import VAMP


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        
        # build
        self.config = config
        self.VAMP = VAMP(config).to(config.device)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = 'Simulations/VAMP/' + f'{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
    @torch.no_grad()
    def run(self, SNRdB: float) -> Loss:
        """_summary_

        Args:
            SNRdB (float): _description_

        Returns:
            Loss: _description_
        """
        SNR = 10 ** ( SNRdB / 10)
        self.channel.generate_channel()
        x = self.data.generate()
        y = self.channel(x, SNR)
        loss = self.VAMP(x, y, self.channel)
        print(SNRdB, loss.SER)
        return loss
    
    def simulate(self, epochs: int, start: float, final: float, step: float = 1):
        """_summary_

        Args:
            epochs (int): _description_
            start (float): _description_
            final (float): _description_
            step (float, optional): _description_. Defaults to 1.
        """
        SNRdB_range = np.arange(start, final+step, step)
        loss = Loss(config)
        for SNRdB in SNRdB_range:
            loss.dump()
            for i in range(epochs):
                print(SNRdB, i)
                loss.collect(self.run(SNRdB))
            loss.average(epochs)
            loss.export(f"SNRdB={str(SNRdB).replace('.', ',')}", self.path)

if __name__ == "__main__":
    Nt = 100
    Na = 5
    Nr = 30
    lb = 20
    lh = 3
    config = Config(batch=100,
                    N_transmit_antenna=Nt,
                    N_active_antenna=Na,
                    N_receive_antenna=Nr,
                    block_length=lb,
                    channel_length=lh,
                    iterations=20,
                    alphabet='OOK')
    model = Model(config)
    model.simulate(100, 10, 20, 1)