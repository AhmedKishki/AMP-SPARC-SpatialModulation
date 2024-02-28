import os
import json

import numpy as np
import matplotlib.pyplot as plt

from config import Config

class Plotter:
    def __init__(self, config: Config, algorithm):
        self.algorithm = algorithm
        self.mode = config.mode
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.Lin, self.Lh = config.Lin, config.Lh
        self.profile = config.profile
        self.iterations = config.N_Layers
        self.alphabet = config.alphabet
        self.name = f'{config.alphabet},Nt={config.Nt},Na={config.Na},Nr={config.Nr},Lh={config.Lh},{config.trunc},{config.mode},{config.profile}'
        self.dir = f'Simulations/{algorithm}/{config.name}'
        self.sim = {}
        
        self.N = 0
        for sim in os.listdir(self.dir):
            if sim.endswith('.json'):
                with open(f'{self.dir}/{sim}', 'r') as json_sim_file:
                    self.sim[self.N] = json.load(json_sim_file)
                    self.N += 1
                    
    def plot_metrics(self):
        EbN0, limit, fer, nMSE, nMSEf, nMSEm, nMSEL, ver, verf, verm, verL, ber, iber, sber, ier, ser, iter = self.get_metrics()
        sort = EbN0.argsort()
        EbN0 = EbN0[sort]
        ver_best = np.min(ver, axis=1)[sort]
        fer_best = np.min(fer, axis=1)[sort]
        nMSE_best = np.min(nMSE, axis=1)[sort]
        ber_best = np.min(ber, axis=1)[sort]
        
        plt.figure(figsize=(8, 6))
        plt.semilogy(EbN0, fer_best, label='FER', color='blue')
        plt.semilogy(EbN0, ver_best, label='SER', color='orange')
        plt.semilogy(EbN0, nMSE_best, label='NMSE', color='red')
        plt.semilogy(EbN0, ber_best, label='BER', color='green')
        plt.axvline(x = limit, color = 'black', label = 'Shannon Limit')
        plt.xlabel('$E_b/N_0$ (dB)')
        plt.ylabel('error rate')
        plt.title(f'{self.alphabet} Nt={self.Nt} Na={self.Na} Nr={self.Nr} L={self.Lin} Lh={self.Lh}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.dir}/{self.name}_plot.png')
        
    def plot_iter(self):
        EbN0, limit, fer, nMSE, nMSEf, nMSEm, nMSEL, ver, verf, verm, verL, ber, iber, sber, ier, ser, iter = self.get_metrics()
        sort = EbN0.argsort()
        EbN0 = EbN0[sort]
        iter = iter[sort]
        plt.figure(figsize=(8, 6))
        plt.plot(EbN0, iter, label='iterations', color='blue')
        plt.axvline(x = limit, color = 'black', label = 'Shannon Limit')
        plt.xlabel('$E_b/N_0$ (dB)') 
        plt.ylabel('Iterations')
        plt.title(f'{self.alphabet} Nt={self.Nt} Na={self.Na} Nr={self.Nr} L={self.Lin} Lh={self.Lh}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.dir}/{self.name}_iter_plot.png')
    
    def get_metrics(self):
        EbN0 = np.zeros(self.N)
        iter = np.zeros(self.N)
        fer = np.zeros((self.N, self.iterations))
        ver = np.zeros((self.N, self.iterations))
        verf = np.zeros((self.N, self.iterations))
        verm = np.zeros((self.N, self.iterations))
        verL = np.zeros((self.N, self.iterations))
        ber = np.zeros((self.N, self.iterations))
        ier = np.zeros((self.N, self.iterations))
        ser = np.zeros((self.N, self.iterations))
        iber = np.zeros((self.N, self.iterations))
        sber = np.zeros((self.N, self.iterations))
        nMSE = np.zeros((self.N, self.iterations))
        nMSEf = np.zeros((self.N, self.iterations))
        nMSEm = np.zeros((self.N, self.iterations))
        nMSEL = np.zeros((self.N, self.iterations))
        
        for i, sim in self.sim.items():
            EbN0[i] = sim['EbN0dB']
            iter[i] = sim['T']
            fer[i] = sim['fer']
            ver[i] = sim['ver']
            verf[i] = sim['verf']
            verm[i] = sim['verm']
            verL[i] = sim['verL']
            ber[i] = sim['ber']
            ier[i] = sim['ier']
            ser[i] = sim['ser']
            iber[i] = sim['iber']
            sber[i] = sim['sber']
            nMSE[i] = sim['nMSE']
            nMSEf[i] = sim['nMSEf']
            nMSEm[i] = sim['nMSEm']
            nMSEL[i] = sim['nMSEL']
        limit = sim['ShannonLimitdB']
            
        return EbN0, limit, fer, nMSE, nMSEf, nMSEm, nMSEL, ver, verf, verm, verL, ber, iber, sber, ier, ser, iter
    
if __name__ == "__main__":
    pass