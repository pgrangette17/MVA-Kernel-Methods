import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

class ShortestPathKernel():

    def __init__(self):
        self.delta = None
        self.phi = None
    
    def Floyd_transformation(self, A):
        n = A.shape[0]
        cost = np.zeros(A.shape)
        for i in range(n):
            for j in range(n):
                if A[i,j] != 0 and i!=j :
                    cost[i,j] = A[i,j]
                elif i==j :
                    cost[i,j] = 0
                else :
                    cost[i,j] = np.inf
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if cost[i,k] + cost[k,i] <cost[i,j]:
                        cost[i,j] = cost[i,k] + cost[k,i]
        return cost
    
    def shortest_paths(self, adjacency_matrix):
        return dijkstra(csr_matrix(adjacency_matrix))
    
    def get_occurrences(self, SP, delta):
        occ = np.zeros(delta+1)
        for val in np.unique(SP):
            if type(val)!=int :
                break
            occ[int(val)] = np.sum(SP == val)
        occ[-1] = np.sum(SP >= delta) 
        return occ

    def fit_transform(self, data):
        delta = 0 # delta corresponds to the maximal nodes in the training dataset +1 (+1 if there is a graph in the test set that is larger)
        for graph in data :
            if graph.number_of_nodes() > delta :
                delta = graph.number_of_nodes()
        self.delta = delta

        phi = np.zeros(shape=(len(data), self.delta+1))
        for i, graph in enumerate(data) : 
            shortest_paths_mat = self.shortest_paths(nx.to_numpy_array(graph))
            phi[i] = self.get_occurrences(shortest_paths_mat, self.delta)
        self.phi_train = phi
        return phi
    
    def transform(self, data):
        phi = np.zeros((len(data), self.delta+1))
        for i, graph in enumerate(data) : 
            shortest_paths_mat = self.shortest_paths(nx.to_numpy_array(graph))
            phi[i] = self.get_occurrences(shortest_paths_mat, self.delta)
        return phi
