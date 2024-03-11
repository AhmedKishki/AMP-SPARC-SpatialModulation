import os

import json
import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from scamp import SCAMP
from plotter import Plotter


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.amp = SCAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/SCAMP/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        W, A = self.channel.generate_as_sparc()
        y = A @ x + self.channel.awgn(SNR)
        loss = self.amp(W, A, y, SNR, x, s, i)
        print(loss.loss['fer'], loss.loss['T'])
        return loss
    
    def simulate(self, epochs: int, final = None, start = None, step: float = 1, res: int = 1):
        if start is None:
            start = int(np.ceil(self.min_snr))
        if final is None:
            final = start + 20.0
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            print(f'EbN0dB={EbN0dB}')
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                if i % res == 0:
                    W, A = self.channel.generate_as_sparc()
                x, s, i = self.data.generate_message()
                y = A @ x + self.channel.awgn(SNR)
                loss = self.amp(W, A, y, SNR, x, s, i)
                self.loss.accumulate(loss)
            self.loss.average(epochs)
            fer = self.loss.loss['fer']
            iter = self.loss.loss['T']
            print(f"FER={fer}, iter={iter}")
            self.loss.export(SNRdB, EbN0dB, self.path)
            if fer < 1e-3:
                break

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Na = 84
    iter = 100
    for trunc in ['tail']:
        for prof in ['uniform']:
            for gen in ['segmented']:
                for Nr in [73]:
                    for alph, Nt in [('OOK', 1344)]:
                        for Lh, Lin in [(6, 32)]:
                            config = Config(
                                            N_transmit_antenna=Nt,
                                            N_active_antenna=Na,
                                            N_receive_antenna=Nr,
                                            block_length=Lin,
                                            channel_length=Lh,
                                            channel_truncation=trunc,
                                            alphabet=alph,
                                            channel_profile=prof,
                                            generator_mode=gen,
                                            batch=1,
                                            iterations=iter,
                                            device=device
                                            )
                            print(config.__dict__)
                            model = Model(config)
                            # model.simulate(epochs=100, step=1., res=10)
                            # model.simulate(epochs=10_000, start=7.0, final=10.0, step=0.25, res=100)
                            Plotter(config, 'VAMP').plot_iter()
                            Plotter(config, 'VAMP').plot_metrics()