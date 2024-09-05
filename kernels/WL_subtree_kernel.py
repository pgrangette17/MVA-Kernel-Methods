import copy 
import numpy as np
import networkx as nx

class WL_subtree_kernel:

    def __init__(self, n_iter=2, n_nodes_labels=50):
        self.n_iter = n_iter
        self.n_nodes_labels = n_nodes_labels
        self.relabeling_dic = None
        self.phi_train = None
        self.phi_shape_iter = list()

    def get_phi_train(self, train_data):
        iter = 0
        input_graphs = copy.deepcopy(train_data)
        phi = list()
        phi_shape_iter = list()

        for i in range(len(input_graphs)):
            graph = copy.copy(input_graphs[i])
            attr, counts = np.unique(list(nx.get_node_attributes(graph, 'labels').values()), return_counts=True)
            phi_graph = np.zeros(self.n_nodes_labels)
            phi_graph[attr] = counts
            phi.append(phi_graph)

        while iter < self.n_iter :
            features_iter = dict()
            counts_list = dict()
            uniques_feat_graphs = dict()
            for i in range(len(input_graphs)):
                graph = copy.copy(input_graphs[i])
                features_graph = list()
                new_attr = dict()
                #asign multiset-label to each node
                for j in range(graph.number_of_nodes()):
                    if iter == 0:
                      feat_node = str(graph.nodes[j]['labels'][0]) + '.'
                    else :
                      feat_node = str(graph.nodes[j]['labels']) + '.'
                    neighbors = [n for n in graph.neighbors(j)]
                    #sorting each multiset
                    if iter ==0:
                      neighbors_attr = sorted([graph.nodes[neighbor]['labels'][0] for neighbor in neighbors])
                    else :
                      neighbors_attr = sorted([graph.nodes[neighbor]['labels'] for neighbor in neighbors])
                    #label compression
                    feat_node += '.'.join(str(attr) for attr in neighbors_attr)
                    features_graph.append(feat_node)
                    new_attr[j] = {'labels':feat_node}
                    if not features_iter.get(feat_node) :
                        features_iter[feat_node] = 1
                #get features by graph
                unique_features_graph, idx_features, counts = np.unique(features_graph, return_index=True, return_counts=True)
                counts_list[i] = counts
                uniques_feat_graphs[i] = unique_features_graph
                nx.set_node_attributes(graph, new_attr)
                input_graphs[i] = graph
            #sort labels of all graph to define the new width of the compressed node labels in phi
            num_phi_iter = len(features_iter) + 1 #add +1 for other features in test graphs that would not have been seen in training graphs
            phi_shape_iter.append(num_phi_iter)
            new_features_sorted = sorted(list(features_iter.keys()))
            relabeling_dic = dict()
            for i, feat in enumerate(new_features_sorted):
                relabeling_dic[feat] = i
            self.relabeling_dic = relabeling_dic
            #relabeling
            for i in range(len(input_graphs)):
                graph = copy.copy(input_graphs[i])
                phi_graph = np.zeros(num_phi_iter)
                counts = counts_list[i]
                uniques_feat_graph = uniques_feat_graphs[i]
                new_features_graph = dict()
                dic_feat_count = {uniques_feat_graph[i]:counts[i] for i in range(counts.shape[0])}
                for j in range(graph.number_of_nodes()):
                    feat_node = graph.nodes[j]['labels']
                    phi_graph[relabeling_dic[feat_node]] += dic_feat_count[feat_node]
                    new_features_graph[j] = {'labels': relabeling_dic[feat_node]}
                phi[i] = np.hstack((phi[i], phi_graph))
                nx.set_node_attributes(graph, new_features_graph)
                input_graphs[i] = graph
            iter += 1
        self.phi_shape_iter = phi_shape_iter
        return phi, input_graphs


    def get_phi_test(self,Y):
        iter = 0
        input_graphs = copy.deepcopy(Y)
        phi = list()

        for i in range(len(input_graphs)):
            graph = copy.copy(input_graphs[i])
            attr, counts = np.unique(list(nx.get_node_attributes(graph, 'labels').values()), return_counts=True)
            phi_graph = np.zeros(self.n_nodes_labels)
            phi_graph[attr] = counts
            phi.append(phi_graph)

        while iter < self.n_iter :
            for i in range(len(input_graphs)):
                graph = copy.copy(input_graphs[i])
                features_graph = list()
                new_attr = dict()
                phi_graph = np.zeros(self.phi_shape_iter[iter])
                #asign multiset-label to each node
                for j in range(graph.number_of_nodes()):
                    if iter == 0:
                      feat_node = str(graph.nodes[j]['labels'][0]) + '.'
                    else :
                      feat_node = str(graph.nodes[j]['labels']) + '.'
                    neighbors = [n for n in graph.neighbors(j)]
                    #sorting each multiset
                    if iter ==0:
                      neighbors_attr = sorted([graph.nodes[neighbor]['labels'][0] for neighbor in neighbors])
                    else :
                      neighbors_attr = sorted([graph.nodes[neighbor]['labels'] for neighbor in neighbors])
                    #label compression
                    feat_node += '.'.join(str(attr) for attr in neighbors_attr)
                    features_graph.append(feat_node)
                    new_attr[j] = {'labels':feat_node}
                #get features by graph
                unique_features_graph, idx_features, counts = np.unique(features_graph, return_index=True, return_counts=True)
                nx.set_node_attributes(graph, new_attr)
                input_graphs[i] = graph
                for j in range(unique_features_graph.shape[0]):
                    if not self.relabeling_dic.get(unique_features_graph[j]):
                        phi_graph[-1] = counts[j]
                    else :
                        phi_graph[self.relabeling_dic[unique_features_graph[j]]] = counts[j]
                phi[i] = np.hstack((phi[i], phi_graph))
            iter += 1
        return phi, input_graphs

    def get_WL_kernel(self, X, Y):
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        return X_arr @ Y_arr.T
