import networkx as nx
import numpy as np
import random as random


class random_graphlets:
    def __init__(self, num=10, m = 4):
        self.num = num  ## how many random graphlet will cover a graph
        self.m = m ## the number of nodes in a graphlet


    def generate_random_graphlets(self,G):
        ## Input G the graph
        graphlets = []
        for i in range(self.num):
            if len(list(G.nodes)) < self.m:
                return self.generate_random_graphlets(G,self.n-1,self.num)
            subgraph_index = random.sample(list(G.nodes()), self.n)
            subgraph = G.subgraph(subgraph_index)
            graphlets.append(subgraph)
        return graphlets ## list of self.num graphlet of size self.m
    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        graphlets_x = []
        graphlets_y = []
        K = np.zeros((N, M))
        for i in range(N):
            graph_i = X[i]
            graphlets_x.append(self.generate_random_graphlets(graph_i))
        for j in range(M):
            graph_j = Y[j]
            graphlets_y.append(self.generate_random_graphlets(graph_j))
        for i in range(N):
            for j in range(M):
                count = 0
                for graphlet_i in graphlets_x[i]:
                    for graphlet_j in graphlets_y[j]:
                        if nx.is_isomorphic(graphlet_j, graphlet_i): ## this step is quite long and we have too many graphs to compare
                            count += 1
                K[i][j] = count
        return  K  ## Matrix of shape NxM