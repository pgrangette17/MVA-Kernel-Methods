import numpy as np
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp

class KernelSVM:
    
    def __init__(self, C, kernel=None, epsilon = 1e-3):
        self.C = C 
        self.kernel = kernel                   
        self.alpha = None 
        self.b = None
        self.support = None
        self.norm_f = None
        self.epsilon = epsilon
        
    def fit(self, X, y):
        n_samples, _ = X.shape

        # Gram matrix
        K = self.kernel(X,X)
        P = matrix(np.outer(y,y) * K)
        q = matrix((-1)*np.ones(n_samples))
        A = matrix(y, (1,n_samples), "d")
        b = matrix(np.zeros(1))

        if self.C is None:
            G = matrix(np.diag((-1)*np.ones(n_samples)))
            h = matrix(np.zeros(n_samples))
        else:
            block_1 = np.diag((-1)*np.ones(n_samples))
            block_2 = np.identity(n_samples)
            G = matrix(np.vstack((block_1, block_2)))
            block_1 = np.zeros(n_samples)
            block_2 = np.ones(n_samples) * self.C
            h = matrix(np.hstack((block_1, block_2)))

        # solve QP problem
        solution = qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        print('ALPHA')
        print(alpha)

        support = alpha > self.epsilon
        ind = np.arange(len(alpha))[support]
        self.alpha = alpha[support]
        self.support = X[support]
        self.support_y = y[support]
        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.support_y[n]
            self.b -= np.sum(self.alpha * self.support_y * K[ind[n],support])
        self.b /= len(self.alpha)

    def separating_function(self, X):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = np.array(X) @ self.support.T
        return K.dot(self.alpha)

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 1*(d+self.b> 0)