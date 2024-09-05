from scipy import optimize
import numpy as np

class KernelRR:
    
    def __init__(self,kernel,lmbda):
        self.lmbda = lmbda                    
        self.kernel = kernel    
        self.alpha = None 
        self.b = None
        self.support = None
        self.type='ridge'
        
    def fit(self, X, y):
        self.support = X

        N = X.shape[0]

        K = self.kernel(X,X)

        def loss(alpha):
            return (K.dot(alpha)-y).T @ (K.dot(alpha)-y) + self.lmbda*N* alpha.T @ K @ alpha

        def grad_loss(alpha):
            return K @ (K @ alpha - y) + self.lmbda*N* K @ alpha  
        
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha))
        
        self.alpha = optRes.x

        self.b = np.mean(y - K @ self.alpha)
        
    ### Implementation of the separting function $f$ 
    def regression_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(x, self.support)
        return K @ self.alpha

    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        return self.regression_function(X)+self.b