import numpy as np

class KLR():
    """Implementation of Kernel Logistic Regression by recursivity with solveKRR"""
    
    def __init__(self, epsilon=None, n_iter=5):
        self.alpha = None
        self.epsilon = epsilon
        self.support = None
        self.n_iter = n_iter
        self.kernel = None
    
    def sigmoid(self, x):
        s = lambda y: 1 / (1+np.exp(-y))
        return s(x)

    def update_W_z(self, K, alpha, y):
        m = K.dot(alpha)
        P = (-1)*self.sigmoid(-y*m)
        W = self.sigmoid(m) * self.sigmoid(-m)
        z = m - P*y / W
        return W, z
    
    def solveWKRR(self, K, W, z, lmbda):
        n = W.shape[0]
        W_half = np.diag(np.power(W, 0.5))
        intermediate_mat = W_half @ K @ W_half + n*lmbda*np.eye(n)
        return W_half @ np.linalg.pinv(intermediate_mat) @ W_half @ z
        
    def fit(self, X, y, lmbda=0, initialisation=None, kernel=None):

        if kernel :
            self.kernel = kernel
            K = kernel(X, X)
        else :
            #apply linear kernel
            K = X @ X.T
        self.support = X
        old_alpha = np.inf
        if initialisation.any() != None :
            alpha = initialisation
        else :
            alpha = np.ones(y.shape[0])
        if self.epsilon : 
            while np.abs(alpha - old_alpha).any() > self.epsilon :
                old_alpha = alpha
                W, z = self.update_W_z(K, alpha, y)
                alpha = self.solveWKRR(K, W, z, lmbda)
        else :
            iter = 0
            while iter < self.n_iter :
                old_alpha = alpha
                W, z = self.update_W_z(K, alpha, y)
                print('W ', W)
                print('z ', z)
                alpha = self.solveWKRR(K, W, z, lmbda)
                iter += 1
        print('ALPHA ', alpha)
        self.alpha = alpha
    
    def predict_proba(self, X) :
        if self.kernel :
            K = self.kernel(X, self.support)
        else :
            K = X @ self.support.T
        preds = K @ self.alpha
        preds = self.sigmoid(preds)
        return preds
    
    def predict(self, X):
        prob = self.predict_proba(X)
        return 1*(prob > 0.5)
    
