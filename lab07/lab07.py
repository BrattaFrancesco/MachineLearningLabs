import sklearn.datasets
import numpy as np
import scipy
import matplotlib.pyplot as pt

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
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

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(DC,NSamples):
    C = ((DC) @ (DC).T) / float(NSamples)
    return C

def withinCovarianceMatrix(Ds):
    matrixSum = np.zeros((Ds[0].shape[0], Ds[0].shape[0]))
    samplesSum = 0
    for Dclass in Ds:
        DC, _ = datasetMean(Dclass)
        C = covarianceMatrix(DC,Dclass.shape[1])
        matrixSum = matrixSum + np.dot(Dclass.shape[1], C)
        samplesSum = samplesSum + Dclass.shape[1]
    
    return matrixSum / float(samplesSum)

def logpdf_GAU_ND(X, mu, C):
    Y = []
    N = X.shape[0]
    logC = np.linalg.slogdet(C)[1]
    invC = np.linalg.inv(C)
    const = N*np.log(2*np.pi)

    for x in X.T:
        x = mcol(x)
        xMinusMu = (x-mu)
        mult = np.dot(np.dot( xMinusMu.T, invC), xMinusMu)[0, 0]

        Y.append(-0.5*(const+logC+mult))
    return np.array(Y)

def loglikelihood(XND, m_ML, C_ML):
    Y = logpdf_GAU_ND(XND, m_ML, C_ML)
    return Y, Y.sum()

def postProb(logS, prior):
    SJoint = np.dot(np.exp(logS), prior)
    SMarginal = vrow(SJoint.sum(0))
    SPost = np.argmax(SJoint/SMarginal, axis=0)
    return SPost

def logPostProb(S, prior):
    logSJoint = S + mcol(np.log(prior))

    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, 0))

    logSPost = logSJoint - logSMarginal

    SPost = np.argmax(np.exp(logSPost), axis=0)
    return SPost, np.exp(logSPost)

def MVGmaxPostProb(DTR, LTR, DTE, prior):
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    C0 = covarianceMatrix(DC0, DC0.shape[1])
    LL0, ll0 = loglikelihood(DTE, mu0, C0)

    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    C1 = covarianceMatrix(DC1, DC1.shape[1])
    LL1, ll1 = loglikelihood(DTE, mu1, C1)

    DC2, mu2 = datasetMean(DTR[:, LTR == 2])
    C2 = covarianceMatrix(DC2, DC2.shape[1])
    LL2, ll2 = loglikelihood(DTE, mu2, C2)

    logS = np.array([LL0, LL1, LL2])

    SPost = postProb(logS, prior)
    return SPost

def tiedCovClass(DTR, LTR, DTE, prior):
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    DC2, mu2 = datasetMean(DTR[:, LTR == 2])
    withinC = withinCovarianceMatrix([DC0, DC1, DC2])
    LL0, _ = loglikelihood(DTE, mu0, withinC)
    LL1, _ = loglikelihood(DTE, mu1, withinC)
    LL2, _ = loglikelihood(DTE, mu2, withinC)

    logS = np.array([LL0, LL1, LL2])
    SPost = postProb(logS, prior)
    return SPost

def confusionMatrix(classLbl, predClassLbl):
    nClasses = np.max(classLbl) + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLbl.size):
        M[predClassLbl[i], classLbl[i]] += 1
    return M

def costMatrix(nClasses):
    diag = np.eye(nClasses)
    return np.ones((nClasses, nClasses)) -diag

def optimalBayes(posteriors, costMatrix):
    expectedBayesCost = posteriors @ costMatrix
    return np.argmin(expectedBayesCost, 0)

def optimalBayesBinary(llr, prior, Cfn, Cfp):
    th = -np.log((prior * Cfn) / ((1-prior) * Cfp))
    return np.int32(llr>th)

def DCFu(prior, Cfn, Cfp, confMatrix):
    Pfn = confMatrix[0,1] / (confMatrix[0,1] + confMatrix[1,1])
    Pfp = confMatrix[1,0] / (confMatrix[0,0] + confMatrix[1,0])
    return (prior * Cfn * Pfn) + ((1-prior) * Cfp * Pfp)

def DCF(prior, Cfn, Cfp, confMatrix):
    Bdummy = np.minimum((prior * Cfn), ((1 - prior) * Cfp))
    return DCFu(prior, Cfn, Cfp, confMatrix) / Bdummy

def DCFMulticlass(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = confusionMatrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / np.min(costMatrix @ mcol(prior_array))
    return bayesError

def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    llrSorted = llr 

    thresholds = np.concatenate([np.array([-np.inf]), llrSorted, np.array([np.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predLbl = np.int32(llr > th)
        confMatrix = confusionMatrix(classLabels, predLbl)
        dcf = DCF(prior, Cfn, Cfp, confMatrix)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin
    
def computePfpPtp(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs

    Pfn = []
    Pfp = []
    thresholds = np.concatenate([np.array([-np.inf]), llrSorted, np.array([np.inf])])
    for th in thresholds:
        predLbl = np.int32(llr > th)
        confMatrix = confusionMatrix(classLabels, predLbl)
        Pfn.append(confMatrix[0,1] / (confMatrix[0,1] + confMatrix[1,1]))
        Pfp.append(confMatrix[1,0] / (confMatrix[0,0] + confMatrix[1,0]))
    return Pfn, Pfp, thresholds
        

if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    ####MVG confusion matrix####
    print("MVG confusion matrix:")
    predLbl = MVGmaxPostProb(DTR, LTR, DTE, 1/3)
    MVGConfMatrix = confusionMatrix(LTE, predLbl)
    print(MVGConfMatrix)

    ####Tied covariance classifier matrix####
    print("Tied covariance classifier matrix:")
    predLbl = tiedCovClass(DTR, LTR, DTE, 1/3)
    tiedConfMatrix = confusionMatrix(LTE, predLbl)
    print(tiedConfMatrix)

    ####Divina Commedia section####
    commedia_ll = np.load('lab07\\solutions\\commedia_ll.npy')
    commedia_labels = np.load('lab07\\solutions\\commedia_labels.npy')
    
    print("Divina commedia conf matrix")
    predLbl, _ = logPostProb(commedia_ll, np.array([1./3., 1./3., 1./3.]))
    commediaConfMatrix = confusionMatrix(commedia_labels, predLbl)
    print(commediaConfMatrix)

    ####Optimal Byes decision: binary task####
    commedia_llr = np.load('lab07\\solutions\\commedia_llr_infpar.npy')
    commedia_labels = np.load('lab07\\solutions\\commedia_labels_infpar.npy')

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        predLbl = optimalBayesBinary(commedia_llr, prior, Cfn, Cfp)  
        ###Evaluation### 
        confMatrix = confusionMatrix(commedia_labels, predLbl)
        print(confMatrix) 
        print("DFCu: ", np.round(DCFu(prior, Cfn, Cfp, confMatrix), 3))
        print("DFC: ", np.round(DCF(prior, Cfn, Cfp, confMatrix), 3))
        print("minDFC: ", np.round(compute_minDCF_binary_slow(commedia_llr, commedia_labels, prior, Cfn, Cfp), 3))
    
    Pfn, Pfp, _ = computePfpPtp(commedia_llr, commedia_labels)
    pt.figure(0)
    pt.plot(np.array(Pfp), 1-np.array(Pfn))
    pt.show()
    
    #Bayes error plot
    effPriorLogOdds = np.linspace(-3, 3, 21) 
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    DCFs = []
    minDCFs = []
    for effPrior in effPriors:
        predLbl = optimalBayesBinary(commedia_llr, effPrior, 1.0, 1.0)  
        confMatrix = confusionMatrix(commedia_labels, predLbl)
        DCFs.append(DCF(effPrior, 1.0, 1.0, confMatrix))
        minDCFs.append(compute_minDCF_binary_slow(commedia_llr, commedia_labels, effPrior, 1.0, 1.0))
    
    pt.plot(effPriorLogOdds, DCFs, label='DCF', color='r')
    pt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='b')
    pt.ylim([0, 1.1])
    pt.xlim([-3, 3])

    commedia_llr = np.load('lab07\\solutions\\commedia_llr_infpar_eps1.npy')
    commedia_labels = np.load('lab07\\solutions\\commedia_labels_infpar_eps1.npy')

    effPriorLogOdds = np.linspace(-3, 3, 21) 
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    DCFs = []
    minDCFs = []
    for effPrior in effPriors:
        predLbl = optimalBayesBinary(commedia_llr, effPrior, 1.0, 1.0)  
        confMatrix = confusionMatrix(commedia_labels, predLbl)
        DCFs.append(DCF(effPrior, 1.0, 1.0, confMatrix))
        minDCFs.append(compute_minDCF_binary_slow(commedia_llr, commedia_labels, effPrior, 1.0, 1.0))
    
    pt.plot(effPriorLogOdds, DCFs, label='DCF eps 1.0', color='y')
    pt.plot(effPriorLogOdds, minDCFs, label='min DCF eps 1.0', color='c')
    pt.ylim([0, 1.1])
    pt.xlim([-3, 3])

    pt.legend()
    pt.show()

    ####Multiclass task####
    commedia_ll = np.load('lab07\\solutions\\commedia_ll.npy')
    commedia_labels = np.load('lab07\\solutions\\commedia_labels.npy')

    multiPrior = np.array([0.3, 0.4, 0.3])
    multiCostMatrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    print()
    print("eps 0.001")
    _, postP = logPostProb(commedia_ll, multiPrior)
    predLbl = optimalBayes(postP, multiCostMatrix)
    print(confusionMatrix(commedia_labels, predLbl))
    print("DCFu: ", DCFMulticlass(predLbl, commedia_labels, multiPrior, multiCostMatrix, False))
    print("DCF: ", DCFMulticlass(predLbl, commedia_labels, multiPrior, multiCostMatrix, True))

    commedia_ll = np.load('lab07\\solutions\\commedia_ll_eps1.npy')
    commedia_labels = np.load('lab07\\solutions\\commedia_labels_eps1.npy')

    print()
    print("eps 1.0")
    multiPrior = np.array([0.3, 0.4, 0.3])
    multiCostMatrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    _, postP = logPostProb(commedia_ll, multiPrior)
    predLbl = optimalBayes(postP, multiCostMatrix)
    print(confusionMatrix(commedia_labels, predLbl))
    print("DCFu: ", DCFMulticlass(predLbl, commedia_labels, multiPrior, multiCostMatrix, False))
    print("DCF: ", DCFMulticlass(predLbl, commedia_labels, multiPrior, multiCostMatrix, True))