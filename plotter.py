import os
import json

import numpy as np
import matplotlib.pyplot as plt

from config import Config

class Plotter:
    def __init__(self, config: Config):
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.Lin, self.Lh = config.Lin, config.Lh
        self.profile = config.profile
        self.iterations = config.N_Layers
        self.dir = f'Simulations/BAMP/{config.name}'
        self.sim = {}
        
        self.N = 0
        for sim in os.listdir(self.dir):
            if sim.endswith('.json'):
                with open(f'{dir}/{sim}', 'r') as json_sim_file:
                    self.sim[self.N] = json.load(json_sim_file)
                    self.N += 1
                    
    def plot_ver(self):
        EbN0, fer, ver, verf, verm, verL = self.get_ver()
        fer_best = np.min(fer, axis=0)
        ver_best = np.min(ver, axis=0)
        fer_best = np.min(fer, axis=0)
        verf_best = np.min(verf, axis=0)
        verm_best = np.min(verm, axis=0)
        verL_best = np.min(verL, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.semilogy(EbN0, ver_best, label='average ver', color='black')
        plt.semilogy(EbN0, fer_best, label='average fer', color='blue')
        plt.semilogy(EbN0, verf_best, label='first ver', color='red', linestle='dashed')
        plt.semilogy(EbN0, verm_best, label='middle ver', color='red', linestyle='dashdot')
        plt.semilogy(EbN0, verL_best, label='last ver', color='red', linestyle='dotted')
        plt.xlabel('EbN0_dB')
        plt.title('ver and fer')
        plt.legend()
        plt.savefig(f'{self.dir}/ver_plot.png')
    
    def plot_mse(self):
        EbN0, mMSE, pMSE, pMSEf, pMSEm, pMSEL = self.get_mse()
        mMSE_best = np.min(mMSE, axis=0)
        pMSE_best = np.min(pMSE, axis=0)
        pMSEf_best = np.min(pMSEf, axis=0)
        pMSEm_best = np.min(pMSEm, axis=0)
        pMSEL_best = np.min(pMSEL, axis=0)
        
    def plot_ber(self):
        EbN0, ber, ier, ser, iber, sber = self.get_ber()
        ber_best = np.min(ber, axis=0)
        iber_best = np.min(iber, axis=0)
        sber_best = np.min(sber, axis=0)
    
    def get_ver(self):
        EbN0 = np.zeros(self.N)
        fer = np.zeros((self.N, self.iterations))
        ver = np.zeros((self.N, self.iterations))
        verf = np.zeros((self.N, self.iterations))
        verm = np.zeros((self.N, self.iterations))
        verL = np.zeros((self.N, self.iterations))
        
        for i, sim in self.sim.items():
            EbN0[i] = sim['EbN0']
            fer[i] = sim['fer']
            ver[i] = sim['ver']
            verf[i] = sim['verf']
            verm[i] = sim['verm']
            verL[i] = sim['verL']
            
        return EbN0, fer, ver, verf, verm, verL
    
    def get_ber(self):
        EbN0 = np.zeros(self.N)
        ber = np.zeros((self.N, self.iterations))
        ier = np.zeros((self.N, self.iterations))
        ser = np.zeros((self.N, self.iterations))
        iber = np.zeros((self.N, self.iterations))
        sber = np.zeros((self.N, self.iterations))
        
        for i, sim in self.sim.items():
            EbN0[i] = sim['EbN0']
            ber[i] = sim['ber']
            ier[i] = sim['ier']
            ser[i] = sim['ser']
            iber[i] = sim['iber']
            sber[i] = sim['sber']
        
        return EbN0, ber, ier, ser, iber, sber 
    
    def get_mse(self):
        EbN0 = np.zeros(self.N)
        mMSE = np.zeros((self.N, self.iterations))
        pMSE = np.zeros((self.N, self.iterations))
        pMSEf = np.zeros((self.N, self.iterations))
        pMSEm = np.zeros((self.N, self.iterations))
        pMSEL = np.zeros((self.N, self.iterations))
        
        for i, sim in self.sim.items():
            EbN0[i] = sim['EbN0']
            mMSE[i] = sim['mMSE']
            pMSE[i] = sim['pMSE']
            pMSEf[i] = sim['pMSEf']
            pMSEm[i] = sim['pMSEm']
            pMSEL[i] = sim['pMSEL']
        
        return EbN0, mMSE, pMSE, pMSEf, pMSEm, pMSEL
        
    
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
                            Plotter(config).plot_ver()