import numpy as np

class KMeans():

    def __init__(self, k, lmbda=0, max_iter=5):
        self.k = k
        self.lmbda = lmbda
        self.max_iter = max_iter
    
    def similarity(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        X_NM = np.tile((X @ X.T).diagonal(), M).reshape((M,N))
        Y_NM = np.tile((Y @ Y.T).diagonal(), N).reshape((N,M))
        similarity = X_NM.T + Y_NM -2 * X @ Y.T
        return  similarity  ## Matrix of shape NxM
    
    def fit(self, X):
        iter = 0
        idx_centers = np.random.randint(X.shape[0], size=(self.k,))
        centers = X[idx_centers]

        while iter < self.max_iter :
            dist = self.similarity(X, centers)
            idx_centers = np.argmin(dist, axis=1)
            clusters = dict()
            for i in range(self.k):
                idx = np.argwhere(idx_centers==i)
                clusters[i] = np.squeeze(X[idx])
                centers[i] = np.mean(clusters[i])
            iter+=1
        self.centers = centers
        self.clusters = clusters
    
    def predict(self, X):
        dist = self.similarity(X, self.centers)
        idx_centers = np.argmin(dist, axis=1)
        return idx_centers
    
    def get_centers(self):
        return self.centers
    
    def get_sigma(self):
        n_feat = self.clusters[0].shape[1]
        sigma = list()
        for i in range(self.k):
            sigma.append(np.cov(self.clusters[i].T) + self.lmbda*np.eye(n_feat))
        return sigma
