import numpy as np
import networkx as nx
import numpy as np
import pickle as pkl
import copy
from cvxopt.solvers import qp
from kernels.WL_subtree_kernel import WL_subtree_kernel
from kernels.WL_edge_kernel import WL_edge_kernel
from kernels.RBF import RBF
from kernels.ShortestPathKernel import ShortestPathKernel
from kernels.random_walk_v2 import RandomWalkKernel

from classifiers.KernelSVM import KernelSVM
from classifiers.LogisticRegression import KLR
from classifiers.GMM import GMM
from classifiers.KernelLogisticRegression import KernelLogisticRegression
from classifiers.KRR import KernelRR

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


import scipy.stats as sp
from scipy.special import expit
import copy


def import_data():
    with open("data/training_data.pkl", "rb") as f:
        training_data = pkl.load(f)
    with open("data/test_data.pkl", "rb") as f:
        test_data = pkl.load(f)
    with open("data/training_labels.pkl", "rb") as f:
        training_labels = pkl.load(f)
    return training_data, test_data, training_labels

def logistic_regression(x, y, iter=100):
    #add a one at the end of each row of x to take the offset into account
    X = np.hstack((x, np.ones((x.shape[0],1))))
    w = np.random.rand(X.shape[1]) * 0.1
    print(w)
    for i in range(iter):        
        eta = expit(X.dot(w))
        dl = X.T.dot(y - eta)
        diag = np.diag(eta*(1-eta))
        hl = -X.T.dot(diag).dot(X)
        w = w - np.linalg.solve(hl,dl)
    return w

if __name__ == '__main__':

    # import data
    print('IMPORT DATA')
    training_data, test_data, training_labels = import_data()
    
    #choose kernel
    print('COMPUTE KERNELS')
    """Choose the kernel you want to use by de-commenting the lines
        by default : the kernel is the Weisfeiler-Lehman edge kernel"""
    
    '''
    print('Compute the Weisfeiler-Lehman subtree kernel')
    WL_model = WL_subtree_kernel(n_iter=2)
    phi_train, new_graphs = WL_model.get_phi_train(training_data)
    phi_test, _ = WL_model.get_phi_test(test_data)
    '''
    
    
    print('Compute the Weisfeiler-Lehman edge kernel')
    WL_model = WL_edge_kernel(n_iter=2)
    phi_train = WL_model.get_phi_train(training_data)
    phi_test = WL_model.get_phi_test(test_data)
    

    '''
    print('Compute the Shortest Path Kernel')
    sp_model = ShortestPathKernel()
    phi_train = sp_model.fit_transform(training_data)
    phi_test = sp_model.transform(test_data)
    '''
    
    print('END OF KERNELS COMPUTATION')

    print('The feature size is :', phi_train[0].shape)

    # for training : split data to get a validation dataset
    phi_tr, phi_val, y_tr, y_val = train_test_split(phi_train, training_labels, test_size=0.2, stratify=training_labels)

    # transform features and labels into arrays
    phi_tr = np.array(phi_tr)
    phi_val = np.array(phi_val)
    y_tr = np.array(y_tr)
    y_val = np.array(y_val)
    phi_train = np.array(phi_train)
    phi_test = np.array(phi_test)
    training_labels = np.array(training_labels)


    # normalize the graph embeddings
    phi_train = phi_train / np.repeat(np.sum(phi_train, axis=1).reshape((-1,1)), phi_train.shape[1], axis = 1)
    phi_test = phi_test / np.repeat(np.sum(phi_test, axis=1).reshape((-1,1)), phi_test.shape[1], axis = 1)
    phi_tr = phi_tr / np.repeat(np.sum(phi_tr, axis=1).reshape((-1,1)), phi_tr.shape[1], axis = 1)
    phi_val = phi_val / np.repeat(np.sum(phi_val, axis=1).reshape((-1,1)), phi_val.shape[1], axis =1)
    
    #apply classifier
    print('APPLY CLASSIFIER')
    """As for the kernel, choose the classifier you want to you
        As a reminder : only the Kernel Logistic Regression works with the WL kernel"""

    #import the RBF kernel class
    rbf = RBF()
    
    '''
    print('Applying the Kernel SVM classifier')
    clf = KernelSVM(C=1, kernel=rbf.kernel)
    clf.fit(phi_tr, y_tr)
    pred_val = clf.separating_function(phi_val)
    y_val_pred = clf.predict(phi_val)
    print('AUC SCORE : ', roc_auc_score(y_val, pred_val))
    '''

    '''
    print('Applying the SVM classifier from sklearn')
    clf = SVC(kernel=rbf.kernel, probability=True)
    clf.fit(phi_tr, y_tr)
    pred_val = clf.predict_proba(phi_val)
    print('AUC SCORE : ', roc_auc_score(y_val, pred_val[:,1]))
    '''

    print('Applying the Kernel Logistic Regression')
    clf = KernelLogisticRegression(kernel=rbf.kernel, sigma=1., max_iteration=1)
    clf.fit(phi_tr, y_tr)
    pred = clf.predict(phi_val)
    print('AUC score :', roc_auc_score(y_val, pred))
    #np.savetxt('results/WL_subtree_2_RBF_KLR.csv', pred)
    

    '''
    print('Applying the Kernel Ridge Regression')
    clf = KernelRR(kernel=rbf.kernel, lmbda=1.)
    clf.fit(phi_tr, y_tr)
    pred = clf.predict(phi_val)
    print('AUC score :', roc_auc_score(y_val, pred))
    '''

    '''
    print('Applying the Kernel Logistic Regression from sklearn')
    clf = LogisticRegression()
    clf.fit(phi_tr, y_tr)
    pred_val = clf.predict(phi_val)
    print('AUC SCORE : ', roc_auc_score(y_val, pred_val))
    '''
    
    '''
    #GMM
    print('Applying GMM')
    clf = GMM(k=2, lmbda=1e-5)
    clf.fit(phi_tr)
    pred_val = clf.predict(phi_val)
    pred_proba = clf.predict_proba(phi_val)
    print('AUC score :', roc_auc_score(y_val, pred_proba))
    #end GMM
    '''

    print('CLASSIFIER APPLIED') 
