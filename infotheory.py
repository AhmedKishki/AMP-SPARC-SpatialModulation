
# Authors: Volker Kuehn

"""
==================================================
some routines for computing entropies, mutual information etc.
==================================================

.. autosummary::
   :toctree: generated/

   mi_dmc              -- computes mutual information of DMC 
   mi_awgn             -- computes mutual information of AWGN with discrete input
   mi_awgn2            -- computes mutual information of AWGN with discrete input
   calc_mi             -- computes variance of AWGN if MI is given or vice versa

"""
import numpy as np


def mi_dmc(pmf_x,pmf_y_x):
    """
    inputs
    --------------------
    pmf_x: input distribution
    pmf_y_x: likelihood function, channel statistics
             rows: y
             cols: x
    
    outputs
    --------------------
    mi: mutual information between input and output of DMC
    """
    
    num_y,num_x = np.shape(pmf_y_x)
    pmf_x = pmf_x.flatten()
  
    pmf_y = pmf_y_x @ pmf_x

    mi = 0
    for x in np.arange(num_x):
        tmp = np.log2(pmf_y_x[:,x]) - np.log2(pmf_y)
#        tmp(isinf(abs(tmp))) = 0;
        mi = mi + np.sum(pmf_y_x[:,x]*pmf_x[x] * tmp)


    return mi
 

    
def mi_awgn(x,pmf_x,snrdB=None,noiseVar=None,N=1000):
    """
    function computes mutual information of discrete-input AWGN channel by 
    numerical integration
    either SNR in dB or noise variance has to be input parameter 
     if both are given, noise variance is preferred and SNR input ignored)
     if SNR is given, noise variance is computed considering averave power of x
    
    inputs
    --------------------
    x :    input alphabet
    pmf_x: input distribution (pmf)
    snrdB: signal-to-noise-ratio in dB per real dimension
            for real-valued alphabets, SNR is twice that of complex-valued alphabets
            real Gaussian distribution uses 2 sigma2N in exponent
            complex Gaussian distribution uses sigma2N in exponent
    noise_var: variance of Gaussian noise per real dimension
    N:     number of points for numerical integration 
    
    outputs
    --------------------
    mi: mutual information between input and output of AWGN channel
    """
    
    if (snrdB.any() != None):
        # average transmit power
        Px = np.sum(np.abs(x)**2 * pmf_x)
    
        # determine variance of AWGN per dimension (real, imag)
        snr = 10**(snrdB/10)
        sigma2N = Px / snr
    elif (noiseVar.any()!=None):
        # determine variance of AWGN per dimension (real, imag)
        sigma2N = noiseVar
    else:
        assert('mi_awgn: neither SNRdB nor noise_var given as inputs!')
            
    if np.isrealobj(x):
        # real gaussian distribution requires 2 sigma_N^2 in exponent
        sigma2N *= 2.0
    sigmaN = np.sqrt(sigma2N)
        
    xmax = np.amax(np.abs(x))

    mi = np.zeros(len(sigma2N))
    
    cntrVAR = 0
    for runVAR in sigma2N:
        # determine grid for numerical integration
        ymax = xmax + 10 * sigmaN[cntrVAR]
        ygrid = np.linspace(-ymax,ymax,N)

        if np.iscomplexobj(x):
            yr,yi = np.meshgrid(ygrid,ygrid)
            y = (yr+1j*yi).flatten()
        else:
            y = ygrid

        
        # determine p(y|x) and p(y)
        pmf_y_x = np.zeros((len(y),len(x)))
        pmf_y = np.zeros(len(y))
        cntrX = 0
        for runx in x:            
            tmp = np.exp(-np.abs(y-runx)**2 / sigma2N[cntrVAR])
            pmf_y_x[:,cntrX] = tmp / np.sum(tmp)
            pmf_y += pmf_y_x[:,cntrX] * pmf_x[cntrX]
            cntrX += 1

        # numerical integration to obtain mutual information 
        log_pmf_y_x = np.zeros(np.shape(pmf_y_x))
        ptr = np.nonzero(pmf_y_x)
        log_pmf_y_x[ptr] = np.log2(pmf_y_x[ptr])
        log_pmf_y = np.zeros(np.shape(pmf_y))
        ptr = np.nonzero(pmf_y)
        log_pmf_y[ptr] = np.log2(pmf_y[ptr])
        
        cntrX = 0
        for runx in x:
            mi[cntrVAR] += np.sum(pmf_y_x[:,cntrX] * (log_pmf_y_x[:,cntrX] - log_pmf_y)) * pmf_x[cntrX]
            cntrX += 1
            
        cntrVAR +=1
    
    return mi
 

def mi_awgn2(x,pmf_x,noise_var,N=1000):
    """
    inputs
    --------------------
    x :    input alphabet
    pmf_x: input distribution (pmf)
    noise_var: variance of Gaussian noise
    N:     number of points for numerical integration 
    
    outputs
    --------------------
    mi: mutual information between input and output 
    """
    
    # determine variance of AWGN per dimension (real, imag)
    sigma2N = noise_var
    if np.isrealobj(x):
        # real gaussian distribution requires 2 sigma_N^2 in exponent
        sigma2N *= 2.0
    sigmaN = np.sqrt(sigma2N)
    
    xmax = np.amax(np.abs(x));

    mi = np.zeros(len(noise_var))
    
    cntrVAR = 0
    for runVAR in noise_var:
        # determine grid for numerical integration
        ymax = xmax + 10 * sigmaN[cntrVAR]
        ygrid = np.linspace(-ymax,ymax,N)

        if np.iscomplexobj(x):
            yr,yi = np.meshgrid(ygrid,ygrid)
            y = (yr+1j*yi).flatten()
        else:
            y = ygrid

        
        # determine p(y|x) and p(y)
        pmf_y_x = np.zeros((len(y),len(x)))
        pmf_y = np.zeros(len(y))
        cntrX = 0
        for runx in x:            
            tmp = np.exp(-np.abs(y-runx)**2 / sigma2N[cntrVAR])
            pmf_y_x[:,cntrX] = tmp / np.sum(tmp)
            pmf_y += pmf_y_x[:,cntrX] * pmf_x[cntrX]
            cntrX += 1

        # numerical integration to obtain mutual information 
        log_pmf_y_x = np.zeros(np.shape(pmf_y_x))
        ptr = np.nonzero(pmf_y_x)
        log_pmf_y_x[ptr] = np.log2(pmf_y_x[ptr])
        log_pmf_y = np.zeros(np.shape(pmf_y))
        ptr = np.nonzero(pmf_y)
        log_pmf_y[ptr] = np.log2(pmf_y[ptr])
        
        cntrX = 0
        for runx in x:
            mi[cntrVAR] += np.sum(pmf_y_x[:,cntrX] * (log_pmf_y_x[:,cntrX] - log_pmf_y)) * pmf_x[cntrX]
            cntrX += 1
            
        cntrVAR +=1
    
    return mi
 

    
def calc_MI(input,mode,N_samples=1000,var_max=50):
    '''
        calculating mutual information of Gaussian distributed LLRs with given variance or
        calculating variance of Gaussian distributed noise for given mutual information
        ------------------------------------------------------------------------
        input parameters:
          input    :  vector containing either variances of Gaussian 
                      distributions or mutual information
          opt      :  parameter deciding output input is requested
                      'MI' : input is variance and output is MI
                      'VAR': input is MI and variances are required
          N_samples:  number of samples representing Gaussian distribution
                      optional, default is N_samples = 1000
          var_max:    scalar defines the maximum variance of Gaussian
                      distribution
                      optional, only required for opt=='VAR'
                      default = 50
        ------------------------------------------------------------------------
        output parameters:
          output   :  either mutual information or variance of Gaussian distribution
        ------------------------------------------------------------------------
    '''

    if (mode=='MI'):  # variances are given
        sigma  = np.sqrt(input)
        output = np.zeros_like(input)

        for run in range(input.shape[0]):
            delta = 10 * sigma[run] / N_samples
            x     = np.arange(-5 * sigma[run],5*sigma[run],delta)

            # mutual entropy of a priori information and encoded data
            output[run] = np.sum(np.exp(-(x-input[run]/2)**2/2/input[run]) * np.log2(1+np.exp(-x)))
            output[run] = 1 - output[run] / np.sqrt(2*np.pi*input[run]) * delta
            
    elif (mode=='VAR'):   # MIs are given
        # defining sufficient range of variances
        sigma2 = np.arange(0.01,var_max)
        sigma  = np.sqrt(sigma2)
        Ia     = np.zeros_like(sigma2)

        for run in np.arange(1,sigma2.shape[0]):
            delta = 10 * sigma[run] / N_samples
            x     = np.arange(-5*sigma[run],5*sigma[run],delta)

            # mutual entropy of a priori information and encoded data
            Ia[run] = np.sum(np.exp(-(x-sigma2[run]/2)**2/2/sigma2[run]) * np.log2(1+np.exp(-x)))
            Ia[run] = 1 - Ia[run] / np.sqrt(2*np.pi*sigma2[run]) * delta

        output = np.interp(input, Ia, sigma2)
    else:
        raise ValueError("wrong parameter mode in {info_theory.py/calc_MI}")
        
    return output


        