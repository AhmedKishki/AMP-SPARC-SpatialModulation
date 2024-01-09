import os

import json
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
        self.rate = config.code_rate
        self.BAMP = BAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/BAMP/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        H = self.channel.generate_channel()
        y = H @ x + self.channel.awgn(SNR)
        loss = self.BAMP(x, y, H, SNR, s, i)
        print(loss.loss)
        return loss
    
    def simulate(self, epochs: int, step: float = 1):
        EbN0 = np.arange(-1, 10+step, step)
        SNRdB_range = EbN0 + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0):
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                print(EbN0dB, i)
                self.loss.accumulate(self.run(SNR))
            self.loss.average(epochs)
            print(self.loss.loss)
            self.loss.export(SNRdB, EbN0dB, self.path)

if __name__ == "__main__":
    Nt = 128
    Nr = 32
    Lin = 20
    for trunc in ['tail', 'trunc']:
        for Lh in [3, 5]:
            for Na in [1, 2, 3, 4]:
                for alph in ['OOK','4ASK','QPSK','8PSK','16PSK','16QAM']:
                    for prof in ['uniform','exponential']:
                        for gen in ['random']:
                            config = Config(
                                            N_transmit_antenna=Nt,
                                            N_active_antenna=Na,
                                            N_receive_antenna=Nr,
                                            block_length=Lin,
                                            channel_length=Lh,
                                            channel_truncation=trunc,
                                            alphabet=alph,
                                            channel_profile=prof,
                                            generator_mode=gen
                                            )
                            print(config.__dict__)
                            model = Model(config)
                            model.simulate(epochs=10, step=1)