import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        X_NM = np.tile((X @ X.T).diagonal(), M).reshape((M,N))
        Y_NM = np.tile((Y @ Y.T).diagonal(), N).reshape((N,M))
        similarity = X_NM.T + Y_NM -2 * X @ Y.T
        G = np.exp(-similarity/(2*self.sigma))
        return  G  ## Matrix of shape NxM