import pandas as pd
import pickle as pkl
import numpy as np

if __name__=='__main__' :
    preds = pd.read_csv('results/WL_subtree_2_RBF_KLR.csv')
    with open("data/test_data.pkl", "rb") as f:
        test_data = pkl.load(f)
    print(preds)
    preds.index.name = 'Id'
    preds.index = np.arange(1, 2001)
    print(preds)
    preds.to_csv('results/WL_subtree_2_RBF_KLR.csv')