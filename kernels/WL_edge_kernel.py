import numpy as np
import copy
import networkx as nx

class WL_edge_kernel():

    def __init__(self, n_iter=2, labeled=True, n_edge_labels=4, n_nodes_labels=50):
        self.n_iter = n_iter
        self.edge_labels = n_edge_labels
        self.n_nodes_labels = n_nodes_labels
        self.labeled = labeled
        self.relabeling_node_dic = list()
        self.relabeling_edge_dic = list()
        self.phi_edges = None
        self.phi_edge_shape_iter = list()
    
    def get_phi_train(self, train_data):
        iter = 0
        input_graphs = copy.deepcopy(train_data)
        if self.labeled :
            phi_edge_shape_iter = list()
            phi_edges = list()
            diff_edges = dict()
            count_edges = list()
            for i in range(len(input_graphs)):
                count_diff_edges = dict()
                graph = copy.copy(input_graphs[i])
                nodes_dic = nx.get_node_attributes(graph, 'labels')
                edges_dic = nx.get_edge_attributes(graph, 'labels')
                for [node1, node2] in edges_dic.keys():
                    lab1 = nodes_dic[node1][0]
                    lab2 = nodes_dic[node2][0]
                    if count_diff_edges.get((lab1, lab2, edges_dic[(node1, node2)][0])):
                        count_diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] += 1   
                    elif count_diff_edges.get((lab2, lab1, edges_dic[(node1, node2)][0])):
                        count_diff_edges[(lab2, lab1, edges_dic[(node1, node2)][0])] += 1
                    else :  
                        count_diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] = 1 
                    if not diff_edges.get((lab1, lab2, edges_dic[(node1, node2)][0])) and not diff_edges.get((lab2, lab1, edges_dic[(node1, node2)][0])):
                        diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] = 1
                count_edges.append(count_diff_edges)
            
            # sort edges features
            edge_features_sorted = sorted(list(diff_edges.keys()))
            relabeling_edge_dic = dict()
            for i, (l1, l2, l) in enumerate(edge_features_sorted):
                if relabeling_edge_dic.get((l1, l2, l)):
                    relabeling_edge_dic[(l1, l2, l)] = i
                else :
                    relabeling_edge_dic[(l2, l1, l)] = i
            phi_edge_shape_iter.append(len(diff_edges) + 1)

            # compute initial features from edges
            for i in range(len(input_graphs)):
                count_edge_graph = count_edges[i]
                phi_edge_graph = np.zeros(len(diff_edges) + 1)
                for (l1, l2, l) in list(count_edge_graph.keys()):
                    if relabeling_edge_dic.get((l1, l2, l)):
                        phi_edge_graph[relabeling_edge_dic[(l1, l2, l)]] = count_edge_graph[(l1, l2, l)]
                    else :
                        phi_edge_graph[relabeling_edge_dic[(l2, l1, l)]] = count_edge_graph[(l1, l2, l)]
                phi_edges.append(phi_edge_graph)
            
            self.relabeling_edge_dic.append(relabeling_edge_dic)

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

                    # get node features by graph
                    unique_features_graph, idx_features, counts = np.unique(features_graph, return_index=True, return_counts=True)
                    counts_list[i] = counts
                    uniques_feat_graphs[i] = unique_features_graph
                    nx.set_node_attributes(graph, new_attr)
                    input_graphs[i] = graph

                #sort labels of all graph to define the new width of the compressed node labels in phi
                new_features_sorted = sorted(list(features_iter.keys()))
                relabeling_node_dic = dict()
                for i, feat in enumerate(new_features_sorted):
                    relabeling_node_dic[feat] = i

                #relabeling
                for i in range(len(input_graphs)):
                    graph = copy.copy(input_graphs[i])
                    counts = counts_list[i]
                    new_features_graph = dict()
                    for j in range(graph.number_of_nodes()):
                        feat_node = graph.nodes[j]['labels']
                        new_features_graph[j] = {'labels': relabeling_node_dic[feat_node]}
                    nx.set_node_attributes(graph, new_features_graph)
                    input_graphs[i] = graph
                
                self.relabeling_node_dic.append(relabeling_node_dic)
                
                # get edge features : first count each feature type
                diff_edges = dict()
                count_edges = list()

                for i in range(len(input_graphs)):
                    count_diff_edges = dict()   
                    nodes_dic = nx.get_node_attributes(graph, 'labels')
                    edges_dic = nx.get_edge_attributes(graph, 'labels')
                    for [node1, node2] in edges_dic.keys():
                        lab1 = nodes_dic[node1]
                        lab2 = nodes_dic[node2]
                        if count_diff_edges.get((lab1, lab2, edges_dic[(node1, node2)][0])):
                            count_diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] += 1   
                        elif count_diff_edges.get((lab2, lab1, edges_dic[(node1, node2)][0])):
                            count_diff_edges[(lab2, lab1, edges_dic[(node1, node2)][0])] += 1
                        else :  
                            count_diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] = 1 
                        if not diff_edges.get((lab1, lab2, edges_dic[(node1, node2)][0])) and not diff_edges.get((lab2, lab1, edges_dic[(node1, node2)][0])):
                            diff_edges[(lab1, lab2, edges_dic[(node1, node2)][0])] = 1
                    count_edges.append(count_diff_edges)
                
                # sort edges features
                edge_features_sorted = sorted(list(diff_edges.keys()))
                relabeling_edge_dic = dict()
                for i, (l1, l2, l) in enumerate(edge_features_sorted):
                    if relabeling_edge_dic.get((l1, l2, l)):
                        relabeling_edge_dic[(l1, l2, l)] = i
                    else :
                        relabeling_edge_dic[(l2, l1, l)] = i
                phi_edge_shape_iter.append(len(diff_edges) + 1)

                # compute initial features from edges
                for i in range(len(input_graphs)):
                    count_edge_graph = count_edges[i]
                    phi_edge_graph = np.zeros(len(diff_edges) + 1)
                    for (l1, l2, l) in list(count_edge_graph.keys()):
                        if relabeling_edge_dic.get((l1, l2, l)):
                            phi_edge_graph[relabeling_edge_dic[(l1, l2, l)]] = count_edge_graph[(l1, l2, l)]
                        else :
                            phi_edge_graph[relabeling_edge_dic[(l2, l1, l)]] = count_edge_graph[(l1, l2, l)]
                    phi_edges[i] = np.hstack((phi_edges[i], phi_edge_graph))
                
                self.relabeling_edge_dic.append(relabeling_edge_dic)
                iter += 1
            self.phi_edge_shape_iter = phi_edge_shape_iter
        return phi_edges
        
    def get_phi_test(self, test_data):
        
        iter = 0
        input_graphs = copy.deepcopy(test_data)
        print(len(input_graphs))
        phi_edges = list()
        relabeling_edge_dic = self.relabeling_edge_dic[0]

        for i in range(len(input_graphs)):
            #compute edge feature
            graph = copy.copy(input_graphs[i])
            phi_edge_graph = np.zeros(self.phi_edge_shape_iter[0])
            nodes_dic = nx.get_node_attributes(graph, 'labels')
            edges_dic = nx.get_edge_attributes(graph, 'labels')
            for [node1, node2] in edges_dic.keys():
                lab1 = nodes_dic[node1][0]
                lab2 = nodes_dic[node2][0]
                if relabeling_edge_dic.get((lab1, lab2, edges_dic[(node1, node2)][0])) :
                    phi_edge_graph[relabeling_edge_dic[(lab1, lab2, edges_dic[(node1, node2)][0])]] += 1
                elif relabeling_edge_dic.get((lab2, lab1, edges_dic[(node1, node2)][0])) :
                    phi_edge_graph[relabeling_edge_dic[(lab2, lab1, edges_dic[(node1, node2)][0])]] += 1
                else :
                    phi_edge_graph[-1] += 1
            phi_edges.append(phi_edge_graph)

        while iter < self.n_iter :
            relabeling_edge_dic = self.relabeling_edge_dic[iter+1]
            relabeling_node_dic = self.relabeling_node_dic[iter]
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
                
                #get node features by graph
                nx.set_node_attributes(graph, new_attr)
                input_graphs[i] = graph

                #get edge features by graph
                phi_edge_graph = np.zeros(self.phi_edge_shape_iter[iter+1])
                nodes_dic = nx.get_node_attributes(graph, 'labels')
                edges_dic = nx.get_edge_attributes(graph, 'labels')
                for [node1, node2] in edges_dic.keys():
                    lab1 = nodes_dic[node1]
                    lab2 = nodes_dic[node2]
                    if relabeling_edge_dic.get((lab1, lab2, edges_dic[(node1, node2)][0])) :
                        phi_edge_graph[relabeling_edge_dic[(lab1, lab2, edges_dic[(node1, node2)][0])]] += 1
                    elif relabeling_edge_dic.get((lab2, lab1, edges_dic[(node1, node2)][0])) :
                        phi_edge_graph[relabeling_edge_dic[(lab2, lab1, edges_dic[(node1, node2)][0])]] += 1
                    else :
                        phi_edge_graph[-1] += 1
                phi_edges[i] = np.hstack((phi_edges[i], phi_edge_graph))
            iter += 1
        return phi_edges