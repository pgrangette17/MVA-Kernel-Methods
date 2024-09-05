import networkx as nx
import numpy as np
import random as random


class similarity_kernel:
    def __init__(self, num_steps=10):
        self.num_steps = num_steps  ##how many transition are allowed
    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        K = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                G = nx.cartesian_product(X[i], Y[i])
                A = nx.adjacency_matrix(G).todense()
                D_inv = np.diagflat([1/val for val in np.sum(A,axis = 1)])
                P = D_inv @ A ##transition matrix
                similarity = np.zeros(P.shape)
                en_cour_mult = np.eye(P.shape[0])
                for t in range(self.num_steps):
                    similarity += en_cour_mult
                    en_cour_mult = en_cour_mult @ P
                K[i][j] = np.trace(similarity)
        return  K  ## Matrix of shape NxM