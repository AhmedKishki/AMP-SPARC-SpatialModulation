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
        self.min_snr_dB = int(np.floor(config.min_snr_dB))
        self.snr_max_dB = self.min_snr_dB + 10
        self.BAMP = BAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = 'Simulations/BAMP/' + f'{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
    @torch.no_grad()
    def run(self, SNRdB: float) -> Loss:
        SNR = 10 ** ( SNRdB / 10)
        self.channel.generate_channel()
        x = self.data.generate()[0]
        y = self.channel(x, SNR)
        loss = self.BAMP(x, y, self.channel)
        print(loss.MSE)
        print(loss.VER)
        print(loss.SER)
        print(loss.FER)
        return loss
    
    def simulate(self, epochs: int, snr_step_dB: float = 1):
        SNRdB_range = np.arange(self.min_snr_dB, self.snr_max_dB+snr_step_dB, snr_step_dB)
        for SNRdB in SNRdB_range:
            self.loss.init()
            for i in range(epochs):
                print(SNRdB, i)
                self.loss.accumulate(self.run(SNRdB))
            self.loss.average(epochs)
            self.loss.export(f"SNRdB={str(SNRdB).replace('.', ',')}", self.path)

if __name__ == "__main__":
    Nt = 100
    Nr = 30
    lb = 20
    lh = 5
    for Na in [5]:
        config = Config(batch=100,
                        N_transmit_antenna=Nt,
                        N_active_antenna=Na,
                        N_receive_antenna=Nr,
                        block_length=lb,
                        channel_length=lh,
                        iterations=20,
                        alphabet='OOK')
        model = Model(config)
        model.simulate(epochs=10, snr_step_dB=1)