import sklearn.datasets as datasets
import numpy as np

def load_iris(): 
    return datasets.load_iris()['data'].T, datasets.load_iris ()['target'] 

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = l[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

if __name__ == "__main__":                                                            
    DIris, LIris = load_iris() 
    D = DIris[:, LIris != 0] 
    L = LIris[LIris != 0]

    