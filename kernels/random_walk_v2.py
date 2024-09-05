import networkx as nx
import numpy as np
import random as random


class RandomWalkKernel:
    def __init__(self,num_paths, path_length):
        self.num_paths = num_paths  ##how many paths are made
        self.path_length = path_length
        self.paths = {}
        
    def random_walk(self, graph, node):
        path = [node]
        for i in range(self.path_length-1):
            neighbors = list(graph.neighbors(path[-1]))
            if len(neighbors) == 0:
                break
            next_node = random.choice(neighbors)
            path.append(next_node)
        return path
    
    def node_similarity(self, G1, G2, n1, n2):
        if G1.nodes[n1]['labels'] ==  G2.nodes[n2]['labels']:
            return 1
        else :
            return 0
        
    def path_similarity(self, G1, G2, path1, path2):
        sim_sum = 0.0
        count = 0
        for n1 in path1:
            for n2 in path2:
                sim = self.node_similarity(G1,G2,n1, n2)
                sim_sum += sim
                count += 1
        if count == 0:
            return 0.0
        else:
            return sim_sum / count

    def compute_paths(self, graph):
        paths = {}
        for node in graph.nodes():
            node_paths = []
            for i in range(self.num_paths):
                node_paths.append(self.random_walk(graph, node))
            paths[node] = node_paths
        return paths
    
    def compute_all_paths(self, graphs):
        for graph in graphs:
            self.paths[graph] = self.compute_paths(graph)
        return list(self.paths.values())
    
    def random_walk_kernel(self, G1, G2):
        similarity = 0
        paths1 = self.paths[G1]
        paths2 = self.paths[G2]
        for node1 in paths1:
            for path1 in paths1[node1]:
                for node2 in paths2:
                    for path2 in paths2[node2]:
                        similarity += self.path_similarity(G1, G2, path1, path2)
        similarity /= (self.num_paths**2) * len(G1) * len(G2)

        return similarity

    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        K = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                K[i,j] = self.random_walk_kernel(X[i], Y[j])
        return K ## Matrix of shape NxM
