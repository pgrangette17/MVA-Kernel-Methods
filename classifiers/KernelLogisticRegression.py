import numpy as np
from copy import deepcopy

class KernelLogisticRegression():
    """Implementation of Kernel Logistic Regression by computing the Newton algorithm
        This is the version of KLR that works."""

    def __init__(self, kernel='gaussian', lmbda=1e-3, max_iteration=1000, sigma=None):
        self.alpha = None
        self.kernel = kernel
        self.max_iteration = max_iteration
        self.lmbda = lmbda
        self.sigma =sigma

    def sigmoid(self, K, alpha):                                                        
        return 1.0 / (1.0 + np.exp(- np.dot(K, alpha) )) 
    
    def gradient(self, K, y, alpha):
        y_0 = np.where(y == 0)
        y_1 = np.where(y == 1)
        n_0 = len(y_0[0])
        n_1 = len(y_1[0])
        p = self.sigmoid(K, alpha)
        term_0 = 1.0/n_0 * np.dot(K[y_0].T, (p[y_0] - np.ones((n_0, 1))))
        term_1 = 1.0/n_1 * np.dot(K[y_1].T, (p[y_1] - np.ones((n_1, 1))))
        return term_0 + term_1 + 2.0 * self.lmbda * K.dot(alpha)
    
    def hessian(self, K, y, alpha):
        y_0 = np.where(y == 0)
        y_1 = np.where(y == 1)
        n_0 = len(y_0[0])
        n_1 = len(y_1[0])
        p = self.sigmoid(K, alpha)
        term_0 = 1.0/n_0 * K[y_0].T.dot(np.diagflat(p[y_0])).dot(np.diagflat(np.ones((n_0, 1)) - p[y_0])).dot(K[y_0])
        term_1 = 1.0/n_1 * K[y_1].T.dot(np.diagflat(p[y_1])).dot(np.diagflat(np.ones((n_1, 1)) - p[y_1])).dot(K[y_1])
        H = term_0 + term_1 + 2.0 * self.lmbda * K
        return H
    
    def Newton(self, K, y_train):
        alpha = np.zeros((len(K), 1))
    
        iteration = 0
        while iteration < self.max_iteration:
            alpha_prev = deepcopy(alpha)
            g = self.gradient(K, y_train, alpha)
            H = self.hessian(K, y_train, alpha)
            alpha = alpha_prev - 0.5*np.linalg.solve(H+self.lmbda*np.eye(H.shape[0]), g)
            if (np.linalg.norm(alpha - alpha_prev) < 10**-2):
                return alpha    
            iteration += 1
            print('Iteration {} : gap : {}'.format(iteration, np.linalg.norm(alpha - alpha_prev)))        
        return alpha
    
    def fit(self, X, y):
        K = self.kernel(X, X)
        self.alpha = self.Newton(K, y)
        self.X = X
    
    def predict(self, X):
        K = self.kernel(X, self.X)
        pred = K @ self.alpha
        return pred-0.5

