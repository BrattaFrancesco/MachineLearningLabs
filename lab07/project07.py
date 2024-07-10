import sklearn.datasets
import numpy as np
import scipy
import matplotlib.pyplot as pt

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

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

    logS = np.array([LL0, LL1])

    SPost = postProb(logS, prior)
    return SPost, LL1-LL0

def NaiveBayesClass(DTR, LTR, DTE, prior):
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    C0 = np.diag(np.diag(covarianceMatrix(DC0, DC0.shape[1])))
    LL0, ll0 = loglikelihood(DTE, mu0, C0)

    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    C1 = np.diag(np.diag(covarianceMatrix(DC1, DC1.shape[1])))
    LL1, ll1 = loglikelihood(DTE, mu1, C1)
    
    logS = np.array([LL0, LL1])
    SPost = postProb(logS, prior)
    return SPost, LL1-LL0

def tiedCovClass(DTR, LTR, DTE, prior):
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    withinC = withinCovarianceMatrix([DC0, DC1])
    LL0, _ = loglikelihood(DTE, mu0, withinC)
    LL1, _ = loglikelihood(DTE, mu1, withinC)

    logS = np.array([LL0, LL1])
    SPost = postProb(logS, prior)
    return SPost, LL1-LL0

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

def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut) # we return also the corresponding thresholds

def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

def PCA(C, m, D):
    #get eigenvalues and eigenvectors, sorted from the smaller to the largest
    _, U = np.linalg.eigh(C)

    #computing SVD
    U, _, _ = np.linalg.svd(C)

    #changing the sign of an eigenvector to flip the image in the scatterplot
    U[:,1] = -U[:,1]
    
    P = U[:, :m]

    DP = np.dot(P.T, D)

    return DP, P

if __name__ == "__main__":
    fileName = "datasets/trainData.csv"

    D, L = load(fileName)
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # Define applications
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]

    effective_priors = [(cfn * pi) / (cfn * pi + cfp * (1 - pi)) for pi, cfn, cfp in applications]

    print(effective_priors)

    """ for (pi, cfn, cfp), eff_prior in zip(applications, effective_priors):
        print("Application: ", (pi, cfn, cfp))
        MVGPost, LLrMVG = MVGmaxPostProb(DTR,LTR,DTE, pi)
        NaivePost, LLrNaive = NaiveBayesClass(DTR,LTR,DTE, pi)
        TiedPost, LLrTied = tiedCovClass(DTR,LTR,DTE,pi)
        MVGBayes = optimalBayesBinary(LLrMVG, pi, cfn, cfp)
        NaiveBayes = optimalBayesBinary(LLrNaive, pi, cfn, cfp)
        TiedBayes = optimalBayesBinary(LLrTied, pi, cfn, cfp)

        confMatrix = confusionMatrix(LTE, MVGBayes)
        print("MVG DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
        print("MVG DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
        print("MVG minDFC: ", np.round(compute_minDCF_binary_fast(LLrMVG, LTE, pi, cfn, cfp), 3))

        confMatrix = confusionMatrix(LTE, NaiveBayes)
        print("Naive DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
        print("Naive DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
        print("Naive minDFC: ", np.round(compute_minDCF_binary_fast(LLrNaive, LTE, pi, cfn, cfp), 3))

        confMatrix = confusionMatrix(LTE, TiedBayes)
        print("Tied DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
        print("Tied DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
        print("Tied minDFC: ", np.round(compute_minDCF_binary_fast(LLrTied, LTE, pi, cfn, cfp), 3))
        print()

        for i in range(1, 7):
            print("PCA m=", i)
            
            DtrC, mu = datasetMean(DTR)
            Ctr = covarianceMatrix(DtrC, DTR.shape[1])
            _, P = PCA(Ctr, i, DTR)
            
            DTR_pca = np.dot( P.T, DTR )
            DTE_pca = np.dot( P.T, DTE )

            MVGPost, LLrMVG = MVGmaxPostProb(DTR_pca,LTR,DTE_pca, pi)
            NaivePost, LLrNaive = NaiveBayesClass(DTR_pca,LTR,DTE_pca, pi)
            TiedPost, LLrTied = tiedCovClass(DTR_pca,LTR,DTE_pca,pi)
            MVGBayes = optimalBayesBinary(LLrMVG, pi, cfn, cfp)
            NaiveBayes = optimalBayesBinary(LLrNaive, pi, cfn, cfp)
            TiedBayes = optimalBayesBinary(LLrTied, pi, cfn, cfp)

            confMatrix = confusionMatrix(LTE, MVGBayes)
            print("MVG DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
            print("MVG DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
            print("MVG minDFC: ", np.round(compute_minDCF_binary_fast(LLrMVG, LTE, pi, cfn, cfp), 3))

            confMatrix = confusionMatrix(LTE, NaiveBayes)
            print("Naive DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
            print("Naive DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
            print("Naive minDFC: ", np.round(compute_minDCF_binary_fast(LLrNaive, LTE, pi, cfn, cfp), 3))

            confMatrix = confusionMatrix(LTE, TiedBayes)
            print("Tied DFCu: ", np.round(DCFu(pi, cfn, cfp, confMatrix), 3))
            print("Tied DFC: ", np.round(DCF(pi, cfn, cfp, confMatrix), 3))
            print("Tied minDFC: ", np.round(compute_minDCF_binary_fast(LLrTied, LTE, pi, cfn, cfp), 3))
            print()
        print() """
    
    #Bayes error plot
    DtrC, mu = datasetMean(DTR)
    Ctr = covarianceMatrix(DtrC, DTR.shape[1])
    _, P = PCA(Ctr, 5, DTR)
            
    DTR_pca = np.dot( P.T, DTR )
    DTE_pca = np.dot( P.T, DTE )

    _, LLrMVG = MVGmaxPostProb(DTR_pca,LTR,DTE_pca, 0.1)
    _, LLrNaive = NaiveBayesClass(DTR_pca,LTR,DTE_pca, 0.1)
    _, LLrTied = tiedCovClass(DTR_pca,LTR,DTE_pca,0.1)
    
    models = [("MVG", LLrMVG), ("Naive", LLrNaive), ("Tied", LLrTied)]

    for model in models:
        effPriorLogOdds = np.linspace(-4, 4, 21) 
        effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
        DCFs = []
        minDCFs = []
        for effPrior in effPriors:
            predLbl = optimalBayesBinary(model[1], effPrior, 1.0, 1.0)  
            confMatrix = confusionMatrix(LTE, predLbl)
            DCFs.append(DCF(effPrior, 1.0, 1.0, confMatrix))
            minDCFs.append(compute_minDCF_binary_fast(LLrMVG, LTE, effPrior, 1.0, 1.0))
        
        pt.plot(effPriorLogOdds, DCFs, label='DCF', color='r')
        pt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='b')
        pt.ylim([0, 1.1])
        pt.xlim([-4, 4])
        pt.title(model[0])
        pt.legend()
        pt.figure()

    pt.show()