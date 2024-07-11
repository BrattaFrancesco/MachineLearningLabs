import numpy as np
import scipy
import scipy.optimize 
import sklearn.datasets
import matplotlib.pyplot as plt

import bayesRisk

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def fun(x):
    f = np.power((x[0]+3), 2) + np.sin(x[0]) + np.power((x[1]+1), 2)
    gradiendY = 2*(x[0]+3) + np.cos(x[0])
    gradiendZ = 2*(x[1]+1)
    return f, np.array([gradiendY, gradiendZ])
           
def load(fileName):
    print("Loading dataset...")
    f=open(fileName, 'r')
    
    datasetList = []
    labelList = []

    line = f.readline()
    while(line != ''):
        fields = line.split(',')
        x = np.array([float(i) for i in fields[0:-1]])
        x = mcol(x)
        l = int(fields[len(fields)-1].replace("\n", ""))
        datasetList.append(x)
        labelList.append(l)
        line = f.readline()

    return np.hstack(datasetList), np.array(labelList)

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

def logisticRegressionClassification(DTR, LTR, DVAL, LVAL):
    errs = []
    minDCFs = []
    actDCFs = []
    lambdas = np.logspace(-4, 2, 13)
    
    for lamb in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, lamb) # Train model
        sVal = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        errs.append(err)
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - np.log(pEmp / (1-pEmp))
        minDCF = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.5: %.4f' % minDCF)
        minDCFs.append(minDCF)
        actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('actDCF - pT = 0.5: %.4f' % actDCF)
        actDCFs.append(actDCF)
    
    plt.plot(lambdas, minDCFs, label='minDCF')
    plt.plot(lambdas, actDCFs, label='actDCF')
    plt.xscale('log', base=10)
    plt.legend()
    plt.show()

def priorWeightedLogisticRegression(DTR, LTR, DVAL, LVAL):
    pT = 0.1
    errs = []
    minDCFs = []
    actDCFs = []
    lambdas = np.logspace(-4, 2, 13)
    
    for lamb in lambdas:
        w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT = pT) # Train model to print the loss
        sVal = np.dot(w.T, DVAL) + b
        sValLLR = sVal - np.log(pT / (1-pT))
        minDCF = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        print ('minDCF - pT = 0.8: %.4f' % minDCF)
        minDCFs.append(minDCF)
        actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
        print ('actDCF - pT = 0.8: %.4f' % actDCF)
        actDCFs.append(actDCF)

    plt.plot(lambdas, minDCFs, label='minDCF')
    plt.plot(lambdas, actDCFs, label='actDCF')
    plt.xscale('log', base=10)
    plt.legend()
    plt.show()

def quadratic_feature_expansion(X):
    X_T = X.T
    X_expanded = []
    for x in X_T:
        outer_product = np.outer(x, x).flatten()
        expanded_feature = np.concatenate([outer_product, x])
        X_expanded.append(expanded_feature)
    X_expanded = np.array(X_expanded).T

    return X_expanded

if __name__== '__main__':
    fileName = "datasets/trainData.csv"

    D, L = load(fileName)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    #full dataset
    logisticRegressionClassification(DTR, LTR, DVAL, LVAL)

    #1/50 dataset
    logisticRegressionClassification(DTR[:, ::50], LTR[::50], DVAL, LVAL)

    #prior weighted
    priorWeightedLogisticRegression(DTR, LTR, DVAL, LVAL)

    _, DTR_mu = datasetMean(DTR)
    DTRC = DTR - DTR_mu
    DVALC = DVAL - DTR_mu

    #full dataset
    logisticRegressionClassification(DTRC, LTR, DVALC, LVAL)

    Dexpanded = quadratic_feature_expansion(D)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(Dexpanded, L)

    #full dataset
    logisticRegressionClassification(DTR, LTR, DVAL, LVAL)
