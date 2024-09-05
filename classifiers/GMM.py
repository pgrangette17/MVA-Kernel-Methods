import numpy as np
from utils.KMeans import KMeans
from scipy.stats import multivariate_normal 

class GMM():

    def __init__(self, k, lmbda=0, max_iter=5):
        self.k = k
        self.lmbda = lmbda
        self.max_iter = int(max_iter)


    def initialisation(self, X):
        self.n = X.shape[0]
        self.phi = np.full(self.k, 1/self.k)
        self.weights = np.full(X.shape, 1/self.k)
        km = KMeans(k=2, lmbda=self.lmbda)
        km.fit(X)
        self.mu = km.get_centers()
        self.sigma = km.get_sigma()
    
    def e_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = np.mean(self.weights, axis=0)
    
    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, i]
            self.mu[i] = np.sum(X * np.repeat(weight, X.shape[1]).reshape((-1, X.shape[1])), axis=0) / weight.sum()
            self.sigma[i] = np.cov(X.T, aweights=(weight/weight.sum()), bias=True)
    
    def fit(self, X):
        self.initialisation(X)
        iter=0
        while iter < self.max_iter :
            self.e_step(X)
            self.m_step(X)
    
    def predict_proba(self, X):
        lkh = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            lkh[:,i] = distribution.pdf(X)
        num = lkh * self.phi
        den = np.sum(num, axis=1)[:, np.newaxis]
        weights = num/den
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
        