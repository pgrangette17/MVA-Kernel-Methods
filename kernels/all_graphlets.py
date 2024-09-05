import networkx as nx
import numpy as np
import random as random
import itertools



class all_graphlets:
    def __init__(self, k = 3):
        self.k = k ## order of graphlets


    def generate_graphlets(self):
        ## nothing
        nodes = list(range(self.k))
        edges = list(itertools.combinations(nodes, 2))

        graphlets = []
        for edge_subset in itertools.chain.from_iterable(itertools.combinations(edges, r) for r in range(len(edges) + 1)):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edge_subset)
            G = nx.freeze(G)
            if nx.is_connected(G) and G not in graphlets:
                graphlets.append(G)

        return graphlets ##output all possible graphlets of order k

    def count_graphlets(self,G,graphlets):
        ##Input graph G and set of graphlets graphlets
        counts = [0] * len(graphlets)
        for nodes in itertools.combinations(G.nodes, len(graphlets[0].nodes)):
            induced_subgraph = G.subgraph(nodes)
            for i, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(induced_subgraph, graphlet):
                    counts[i] += 1
        return counts
    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        graphlets= self.generate_graphlets()
        K = np.zeros((N, M))
        vect_x = []
        vect_y = []
        for i in range(N):
            graph_i = X[i]
            vect_x.append(self.count_graphlets(graph_i,graphlets))
        for j in range(M):
            graph_j = Y[j]
            vect_y.append(self.count_graphlets(graph_j,graphlets))
        for i in range(N):
            for j in range(M):
                K[i][j] = np.dot(vect_x[i],vect_y[j])
        return  K  ## Matrix of shape NxM