import numpy as np
import scipy
import scipy.optimize 
import sklearn.datasets

import bayesRisk

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def fun(x):
    f = np.power((x[0]+3), 2) + np.sin(x[0]) + np.power((x[1]+1), 2)
    gradiendY = 2*(x[0]+3) + np.cos(x[0])
    gradiendZ = 2*(x[1]+1)
    return f, np.array([gradiendY, gradiendZ])
           
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0

    def logreg_obj_with_grad(v):
        w, b = v[0:-1], v[-1]
    
        S = np.dot(mcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * S)

        G = -ZTR / (1.0 + np.exp(ZTR * S))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(mcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

if __name__ == "__main__":
    #x, f, d = scipy.optimize.fmin_l_bfgs_b(fun, np.array([[0],[0]]))
    
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for lamb in [1e-3, 1e-1, 1.0]:
        w, b = trainLogRegBinary(DTR, LTR, lamb) # Train model
        sVal = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - np.log(pEmp / (1-pEmp))
        # Compute optimal decisions for the three priors 0.1, 0.5, 0.9
        print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))
        print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))

        pT = 0.8
        w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT = pT) # Train model to print the loss
        sVal = np.dot(w.T, DVAL) + b
        sValLLR = sVal - np.log(pT / (1-pT))
        print ('minDCF - pT = 0.8: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))
        print ('actDCF - pT = 0.8: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0))
        
        print ()