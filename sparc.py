import numpy as np

class SPARC:
    def __init__(self, Nt, Nr, Lt, Lh, K=1, modulation='OOK', profile='uniform'):
        """

        Args:
            Nt (_type_): _description_
            Nr (_type_): _description_
            L (_type_): _description_
            Lh (_type_): _description_
            K (int, optional): _description_. Defaults to 1.
            modulation (str, optional): _description_. Defaults to 'OOK'.
        """
        self.Nt, self.Nr, self.Lt, self.Lh, self.K = Nt, Nr, Lt, Lh, K
        self.modulation = modulation
        self.Lr = Lt + Lh - 1
        self.n = self.Lr * self.Nr
        self.rate = self.Lt * np.log(self.Nt * self.K) / self.n
        self.kappa = self.Lr / self.Lt
        self.EbN0_min = 10*np.log10((2**self.rate - 1)/self.rate)
        
        self.Ps = 1 / self.Nt
        self.P0 = 1 - self.Ps
        
        self.dtype = np.complex64
        
        if self.modulation == 'OOK' or self.K == 1:
            self.symbols = np.array([1. + 1j*0.])
        
        elif self.modulation == 'PSK':
            self.symbols = np.exp(2j * np.pi * np.arange(self.K) / self.K)
        
        elif self.modulation == 'QAM':
            if self.K == 4:
                pass
            
            elif self.K == 8:
                pass
            
            elif self.K == 16:
                pass
            
        if profile == 'uniform':
            self.pdp = np.ones(self.Lh)
        
        elif profile == 'exponential':
            self.pdp = np.exp(-np.arange(self.Lh)) / np.exp(-np.arange(self.Lh)).mean()
    
    def channel_matrix(self, W: np.ndarray):
        """_summary_

        Args:
            power_allocation (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        hr = np.random.normal(size=(self.Nr, self.Nt, self.Lh))
        hj = np.random.normal(size=(self.Nr, self.Nt, self.Lh))
        h = (hr + 1j * hj) / np.sqrt(2*self.Lt)
        A = np.zeros((self.n, self.Nt*self.Lt), dtype=self.dtype)
        for l in range(self.Lh):
            A += np.kron(np.eye(self.Lr, self.Lt, -l) * np.sqrt(W), h[:,:,l])
        return A
    
    def base_matrix(self, power_allocation=None):
        """
        The mean of any base matrix should equal the signal power, here the signal power is normalized
        to equal one

        Args:
            power (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if power_allocation is None:
            power_allocation = np.ones(self.Lt)
            
        W = np.zeros((self.Lr, self.Lt))
        for l in np.arange(self.Lh):
            W += np.eye(self.Lr, self.Lt, -l) * self.pdp[l] * self.Lr / self.Lh
        return W * power_allocation / power_allocation.mean()
    
    def awgn(self, sigma2: float):
        nr = np.random.normal(size=(self.n, 1)) * np.sqrt(sigma2 / 2)
        nj = np.random.normal(size=(self.n, 1)) * np.sqrt(sigma2 / 2)
        return nr + 1j * nj
    
    def capacity(self, snr: float):
        return np.log(1 + snr)
    
    def exp_power_allocation(self, snr: float, f=None, a=None):
        C = self.capacity(snr)
        if f is None and a is None:
            pa = np.array([2**(-2*C*l/self.Lt) for l in range(self.Lt)])
        elif 0<f and f<1 and a is not None:
            pa1 = np.array([2**(-2*a*C*l/self.Lt) for l in range(int(f*self.Lt))])
            pa2 = np.array([2**(-2*a*C*f) for _ in range(int(f*self.Lt), self.Lt)])
            pa = np.concatenate((pa1, pa2))
        else:
            raise Exception('sth bad hpn')
        return pa
    
    def iter_power_allocation(self):
        pass
    
    def message(self):
        b = np.zeros((self.Lt, self.Nt), dtype=self.dtype)
        index = np.random.randint(self.Nt, size=self.Lt)
        value = np.random.randint(self.K, size=self.Lt)
        for i in range(self.Lt):
            b[i, index[i]] = self.modulate(value[i])
        return b.reshape(-1, 1)
    
    def modulate(self, k):
        """_summary_

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.symbols[k]
    
    def BAMP(self, A: np.ndarray, y: np.ndarray, sigma2: np.ndarray, iterations: int):
        adj = A.conjugate().transpose()
        abs2 = np.abs(A)**2
        xamp = np.zeros((iterations+1, self.Lt*self.Nt, 1), dtype=np.complex64)
        var = np.ones((self.Nt*self.Lt, 1), dtype=np.float32)
        V = np.zeros_like(y)
        Z = y
        for t in range(iterations):
            V_prev, V = V, abs2 @ var
            Z = A @ xamp[t] - V * (y - Z) / (V_prev + sigma2)
            U = 1 / (V + sigma2)
            cov = 1 / (abs2.T @ U)
            r = xamp[t] + cov * (adj @ ((y - Z) * U))
            xamp[t+1], var = self.shrinkage(r, cov)
        return xamp[-1], xamp
    
    def shrinkage(self, r: np.ndarray, cov: np.ndarray):
        G = lambda s: np.exp(- np.abs(r - s)**2 / cov )
        G0, Gs = G(0), G(self.symbols)
        norm = self.regularize(self.P0 * G0 + self.Ps * np.sum(Gs, axis=-1, keepdims=True))
        exp = self.Ps * np.sum(self.symbols * Gs, axis=-1, keepdims=True) / norm
        var = self.Ps * np.sum((self.symbols**2) * Gs, axis=-1, keepdims=True) / norm - np.abs(exp)**2
        return exp, var
    
    def SCAMP(self, 
            W: np.ndarray, 
            A: np.ndarray, 
            y: np.ndarray, 
            sigma2: float, 
            iterations: int):
        """_summary_

        Args:
            W (np.ndarray): _description_
            A (np.ndarray): _description_
            y (np.ndarray): _description_
            sigma2 (float): _description_
            n_iterations (int): _description_
        """
        xamp = np.zeros((iterations+1, self.Lt * self.Nt, 1), dtype=self.dtype)
        phi = np.ones(self.Lr) * np.inf
        z = y
        for t in range(iterations):
            gma = W @ (1 - np.mean(np.abs(xamp[t].reshape((-1, self.Nt)))**2, axis=1))
            v = self.repeat_vector(gma / phi, self.Nr) / self.Lt
            phi = sigma2 + gma
            tau = self.Lr * self.rate / (W.T @ (1 / phi)) / np.log(self.K * self.Nt) / 2
            Q = self.repeat_matrix(np.outer(1/phi, tau), self.Nr, self.Nt)
            z = y - A @ xamp[t] + v.reshape(-1, 1) * z
            xamp[t+1] = self.denoiser(xamp[t] + np.transpose(Q * A).conjugate() @ z, tau)
        return xamp[-1], xamp
    
    def estSCAMP(self, 
            W: np.ndarray, 
            A: np.ndarray, 
            y: np.ndarray, 
            iterations: int):
        """_summary_

        Args:
            W (np.ndarray): _description_
            A (np.ndarray): _description_
            y (np.ndarray): _description_
            n_iterations (int): _description_
        """
        xamp = np.zeros((iterations+1, self.Lt * self.Nt, 1), dtype=self.dtype)
        phi = np.ones(self.Lr) * np.inf
        z = y
        for t in range(iterations):
            gma = W @ (1 - np.mean(np.abs(xamp[t].reshape((-1, self.Nt)))**2, axis=1))
            v = self.repeat_vector(gma / phi, self.Nr)
            phi = np.mean(np.abs(z.reshape((-1, self.Nr)))**2, axis=1)
            tau = self.Lr * self.rate / (W.T @ (1 / phi)) / np.log(self.K * self.Nt) / 2
            Q = self.repeat_matrix(np.outer(1/phi, tau), self.Nr, self.Nt)
            z = y - A @ xamp[t] + v.reshape(-1, 1) * z
            xamp[t+1] = self.denoiser(xamp[t] + np.transpose(Q * A).conjugate() @ z, tau)
        return xamp[-1], xamp
    
    def repeat_vector(self, v: np.ndarray, r: int) -> np.ndarray:
        """
        repeats each entry r times

        Args:
            v_ (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return v.repeat(r)
    
    def repeat_matrix(self, M: np.ndarray, r: int, t: int) -> np.ndarray:
        """
        repeats each entry in an r x t matrix

        Args:
            S_ (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return M.repeat(r, axis=0).repeat(t, axis=1)
        
    def denoiser(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            w (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        s = s.reshape((-1, self.Nt, 1))
        tau = self.repeat_vector(tau, self.Nt).reshape((-1, self.Nt, 1))
        x = np.tile(s / tau, (1, 1, self.K))
        eta = np.exp(self.expthreshold(np.real(x*self.symbols.conjugate())))
        eta2 = self.symbols * eta
        xamp = eta2.sum(axis=2) / eta.sum(axis=2).sum(axis=1, keepdims=True)
        return xamp.reshape(-1, 1)
    
    def expthreshold(self, arg: np.ndarray):
        max = np.log(np.finfo(self.dtype).max) - 1
        arg[arg>max] = max
        return arg
    
    def regularize(self, a):
        a[a==0] = 1e-20
        return a
    
    def hard_decision(self, xamp: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            xamp (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        xamp = xamp.reshape(-1, self.Nt)
        xhat = np.zeros_like(xamp)
        for i in range(self.Lt):
            index = np.argmax(np.abs(xamp[i])**2)
            value = xamp[i, index]
            d = np.inf
            for s in self.symbols:
                ds = np.abs(s - value)**2
                if ds < d:
                    xhat[i, index] = s
                    d = ds
                    if ds == 0:
                        break
        return xhat.reshape(-1, 1)
    
    def bit_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        xhat = xhat.reshape((-1, self.Nt))
        x = x.reshape((-1, self.Nt))
        ber = 0
        for i in range(self.Lt):
            ihat, itrue = np.argwhere(xhat[i]), np.argwhere(x[i])
            if ihat != itrue:
                ber += np.log2(self.Nt) / 2
            if xhat[i, ihat] != x[i, itrue]:
                ber += np.log2(self.K) / 2
        return ber
            
    def section_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        xhat = xhat.reshape(-1, self.Nt)
        x = x.reshape(-1, self.Nt)
        ser = 0
        for i in range(self.Lt):
            if np.count_nonzero(xhat[i] - x[i]) > 0:
                section_index = i
                ser += 1
        ser = ser / self.Lt
        return ser
    
    def frame_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        fer = int(np.count_nonzero(xhat - x) > 0)
        return fer
    
    def normalized_MSE(self, xhat: np.ndarray, x: np.ndarray) -> float:
        nmse = np.mean(np.abs(xhat - x)**2)
        return nmse
    
    def error_rate(self, xhat: np.ndarray, x: np.ndarray) -> dict:
        return {'ser'  : self.section_error_rate(xhat, x),
                'fer'  : self.frame_error_rate(xhat, x),
                'nmmse': self.normalized_MSE(xhat, x)} 
    
    def run(self, EbN0_dB, iterations=20):
        SNRdB = EbN0_dB + 10*np.log10(self.rate)
        snr = 10 ** ( SNRdB / 10)
        C = np.log2(1 + snr)
        # pa = self.exp_power_allocation(snr)
        # #pa = np.ones(self.Lt)
        W = self.base_matrix()
        A = self.channel_matrix(W)
        x = self.message()
        y = A @ x + self.awgn(1/snr)
        xamp_final, xamp = self.SCAMP(W, A, y, 1/snr, iterations)
        xhat = self.hard_decision(xamp_final)
        ser = self.section_error_rate(xhat, x)
        fer = self.frame_error_rate(xhat, x)
        nmse = self.normalized_MSE(xhat, x)
        return ser, fer, nmse, C, self.rate


class SPARC2:
    def __init__(self, Nt, Na, Nr, Lt, Lh, K=1, modulation='OOK', profile='uniform') -> None:
        self.Nt, self.Na, self.Nr = Nt, Na, Nr
        self.Lt, self.Lh, self.K = Lt, Lh, K
        self.Lr = self.Lt + self.Lh -1
        self.n = self.Lr * self.Nr
        self.Ns = self.Na * self.Lt
        self.modulation = modulation
        
        self.Ps = self.Na / self.Nt / self.K
        self.P0 = 1 - self.Na / self.Nt
        
        self.section = self.Nt // self.Na
        
        self.sparc = SPARC(Nt, Nr, Lt, Lh, K, modulation, profile)
        self.symbols = self.sparc.symbols
        self.symbols2 = np.abs(self.symbols)**2
        self.rate = self.Lt * self.Na * np.log(self.Nt * self.K / self.Na) / self.n
        
    def message(self):
        b = np.zeros((self.Lt, self.Nt), dtype=np.complex64)      
        for j in range(self.Lt):
            space_index = [i*self.section + np.random.choice(self.section) for i in range(self.Na)]
            mod_index = np.random.choice(self.K)
            b[j, space_index] = self.symbols[mod_index]
        return b.reshape((-1, 1))
        
    def awgn(self, sigma2: float):
        nr = np.random.normal(size=(self.n, 1)) * np.sqrt(sigma2 / 2)
        nj = np.random.normal(size=(self.n, 1)) * np.sqrt(sigma2 / 2)
        return nr + 1j * nj
    
    def BAMP(self, A: np.ndarray, y: np.ndarray, sigma2: np.ndarray, iterations=20):
        adj = A.conjugate().transpose()
        abs2 = np.abs(A)**2
        xamp = np.zeros((iterations+1, self.Lt*self.Nt, 1), dtype=np.complex64)
        var = np.ones((self.Nt*self.Lt, 1), dtype=np.float32)
        V = np.zeros_like(y)
        Z = y
        for t in range(iterations):
            V_prev, V = V, abs2 @ var
            Z = A @ xamp[t] - V * (y - Z) / (V_prev + sigma2)
            U = 1 / (V + sigma2)
            cov = 1 / (abs2.T @ U)
            r = xamp[t] + cov * (adj @ ((y - Z) * U))
            xamp[t+1], var = self.shrinkage(r, cov)
        return xamp[-1], xamp
    
    def shrinkage(self, r: np.ndarray, cov: np.ndarray):
        G = lambda s: np.exp(- np.abs(r - s)**2 / cov )
        G0, Gs = G(0), G(self.symbols)
        norm = self.regularize(self.P0 * G0 + self.Ps * np.sum(Gs, axis=-1, keepdims=True))
        exp = self.Ps * np.sum(self.symbols * Gs, axis=-1, keepdims=True) / norm
        var = self.Ps * np.sum(self.symbols2 * Gs, axis=-1, keepdims=True) / norm - np.abs(exp)**2
        return exp, var
    
    def regularize(self, a):
        a[a==0] = 1e-20
        return a
    
    def bit_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        xhat = xhat.reshape((-1, self.Nt))
        x = x.reshape((-1, self.Nt))
        ber = 0
        for i in range(self.Lt):
            ihat, itrue = np.argwhere(xhat[i]), np.argwhere(x[i])
            if ihat != itrue:
                ber += np.log2(self.Nt) / 2
            if xhat[i, ihat] != x[i, itrue]:
                ber += np.log2(self.K) / 2
        return ber
            
    def section_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        xhat = xhat.reshape(-1, self.Nt)
        x = x.reshape(-1, self.Nt)
        ser = 0
        for i in range(self.Lt):
            if np.count_nonzero(xhat[i] - x[i]) > 0:
                section_index = i
                ser += 1
        ser = ser / self.Lt
        return ser
    
    def frame_error_rate(self, xhat: np.ndarray, x: np.ndarray) -> float:
        fer = int(np.count_nonzero(xhat - x) > 0)
        return fer
    
    def normalized_MSE(self, xhat: np.ndarray, x: np.ndarray) -> float:
        nmse = np.mean(np.abs(xhat - x)**2)
        return nmse
    
    def error_rate(self, xhat: np.ndarray, x: np.ndarray) -> dict:
        return {'ser'  : self.section_error_rate(xhat, x),
                'fer'  : self.frame_error_rate(xhat, x),
                'nmmse': self.normalized_MSE(xhat, x)}
        
    def hard_decision(self, xamp: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            xamp (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray]: _description_
        """
        xamp = xamp.ravel()
        index = np.abs(xamp).argsort()[-self.Ns:]
        xhat = np.zeros_like(xamp)
        for j, xs in enumerate(xamp[index]):
            d = np.inf
            for i, s in enumerate(self.symbols):
                ds = np.abs(xs - s)
                if ds < d:
                    d = ds
                    xhat[index[j]] = s
                    if ds == 0:
                        break
        return xhat.reshape((-1, 1))
        
    def run(self, EbN0_dB, iterations=20):    
        SNRdB = EbN0_dB + 10*np.log10(self.rate)
        snr = 10 ** ( SNRdB / 10)
        sigma2 = self.Na / snr
        
        C = np.log2(1 + snr)
        
        x = self.message()
        W = self.sparc.base_matrix()
        H = self.sparc.channel_matrix(W)
        print((H@x).var())
        y = H @ x + self.awgn(sigma2)
        xscamp, _ = self.sparc.SCAMP(W, H, y, sigma2/self.Na, iterations)
        xhat1 = self.hard_decision(xscamp)
        
        xbamp, _ = self.BAMP(H, y, sigma2, iterations)
        xhat2 = self.hard_decision(xbamp)
        
        error_scamp = self.error_rate(xhat1, x)
        error_bamp = self.error_rate(xhat2, x)
        
        return error_scamp, error_bamp 
          
        

if __name__ == "__main__":
    print(SPARC2(128, 1, 32, 20, 3, K=4, modulation='PSK', profile='uniform').run(3))