import sklearn.datasets
import numpy as np
import scipy

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
    """ logSJoint_solution = np.load('lab05/solutions/logSJoint_MVG.npy')
    print(np.round(logSJoint-logSJoint_solution)) """

    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    """ logSMarginal_solution = np.load('lab05/solutions/logMarginal_MVG.npy')
    print(np.round(logSMarginal-logSMarginal_solution)) """

    logSPost = logSJoint - logSMarginal
    """ logSPost_solution = np.load('lab05/solutions/logPosterior_MVG.npy')
    print(np.round(logSPost-logSPost_solution)) """

    SPost = np.argmax(np.exp(logSPost), axis=0)
    return SPost

def validate(SPost, LTE):
    acc = ((SPost == LTE).sum()/LTE.shape)[0]
    err = 1-acc
    return acc, err

if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    #####Multivariate Gaussian Classifier#####
    #################Part 1###################
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
    prior = 1/3

    print("Multivariate Gaussian Classifier")
    SPost = postProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(densities): ", np.round(err*100), "%")

    #################Part 2###################
    SPost = logPostProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(log-densities): ", np.round(err*100), "%")

    #####Naive Bayes Gaussian Classifier######
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    C0 = np.diag(np.diag(covarianceMatrix(DC0, DC0.shape[1])))
    LL0, ll0 = loglikelihood(DTE, mu0, C0)

    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    C1 = np.diag(np.diag(covarianceMatrix(DC1, DC1.shape[1])))
    LL1, ll1 = loglikelihood(DTE, mu1, C1)

    DC2, mu2 = datasetMean(DTR[:, LTR == 2])
    C2 = np.diag(np.diag(covarianceMatrix(DC2, DC2.shape[1])))
    LL2, ll2 = loglikelihood(DTE, mu2, C2)
    
    print("Naive Bayes Gaussian Classifier")
    logS = np.array([LL0, LL1, LL2])
    SPost = postProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(densities): ", np.round(err*100), "%")

    SPost = logPostProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(log-densities): ", np.round(err*100), "%")

    #####Tied Covariance Gaussian Classifier#####
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    DC2, mu2 = datasetMean(DTR[:, LTR == 2])
    withinC = withinCovarianceMatrix([DC0, DC1, DC2])
    print(withinC)
    LL0, ll0 = loglikelihood(DTE, mu0, withinC)
    LL1, ll1 = loglikelihood(DTE, mu1, withinC)
    LL2, ll2 = loglikelihood(DTE, mu2, withinC)

    print("Tied Covariance Gaussian Classifier")
    logS = np.array([LL0, LL1, LL2])
    SPost = postProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(densities): ", np.round(err*100), "%")

    SPost = logPostProb(logS, prior)
    acc, err = validate(SPost, LTE)
    print("\tError rate(log-densities): ", np.round(err*100), "%")