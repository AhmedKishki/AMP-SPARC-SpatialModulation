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
from plotter import Plotter


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.amp = BAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/BAMPfinal/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        H = self.channel.generate_channel()
        y = H @ x + self.channel.awgn(SNR)
        loss = self.amp(x, y, H, SNR, s, i)
        print(loss.loss)
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
                    _, A = self.channel.generate_as_sparc()
                x, s, i = self.data.generate_message()
                y = A @ x + self.channel.awgn(SNR)
                loss = self.amp(A, y, SNR, x, s, i)
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
    Na = 16
    iter = 100
    for trunc in ['tail']:
        for prof in ['uniform']:
            for gen in ['segmented']:
                for Nr in [32]:
                    for alph, Nt in [('QPSK', 128), ('BPSK', 256), ('OOK', 512)]:
                        for Lh, Lin in [(6, 25), (9, 40), (3, 10)]:
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
                            model.simulate(epochs=100, start=6, step=1.0, res=2)
                            model.simulate(epochs=10_000, start=6.0, final=10.0, step=0.25, res=100)
                            Plotter(config, 'BAMPfinal').plot_iter()
                            Plotter(config, 'BAMPfinal').plot_metrics()