import numpy as np
import scipy
import scipy.linalg as lalg
import matplotlib.pyplot as plt

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

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

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

def betweenCovarianceMatrix(Ds, mu):
    matrixSum = np.zeros((Ds[0].shape[0], 1))
    sampleSum = 0
    for Dclass in Ds:
        _, classMean = datasetMean(Dclass)
        x = np.dot((classMean - mu), (classMean - mu).T)
        matrixSum = matrixSum + np.dot(Dclass.shape[1], x)
        sampleSum = sampleSum + Dclass.shape[1]
    return matrixSum / float(sampleSum)

def generalizedEigvalLDA(Sb, Sw, m, D):
    #solving the generalized eigenvalue problem to find W (matrix in which columns are the directions of LDA)
    _, U = lalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    #finding basis U for the subspace spanned by W
    UW, _, _ = lalg.svd(W) 
    U = UW[:, 0:m]

    #calculating LDA for the givend dataset
    LDA = np.dot(W.T, D)

    return W, U, LDA

def LDAClassificator(DTR, DVAL, LTR, LVAL):
    threshold = (DTR[0, LTR == 0].mean() + DTR[0, LTR == 1].mean()) / 2.0 # We only have one dimension in the projected Dataset
    
    PredVal = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PredVal[DVAL[0] >= threshold] = 1
    PredVal[DVAL[0] < threshold] = 0

    return PredVal

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
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.argmax(np.exp(logSPost), axis=0)
    return SPost

def validate(pred, LTE):
    acc = ((pred == LTE).sum()/LTE.shape)[0]
    err = 1-acc
    return acc, err

def classification(DTR, LTR, DTE, LTE, startingFeature, endingFeature):
    startingFeature = startingFeature - 1
    DTR = DTR[startingFeature:endingFeature]
    DTE = DTE[startingFeature:endingFeature]

    print("\n################################\nStarting classification...\nFeature from ", startingFeature + 1, "to", endingFeature,
          "\n################################\n")
    #################MVG#####################
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    C0 = covarianceMatrix(DC0, DC0.shape[1])
    LL0, _ = loglikelihood(DTE, mu0, C0)

    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    C1 = covarianceMatrix(DC1, DC1.shape[1])
    LL1, _ = loglikelihood(DTE, mu1, C1)

    llRatio = LL1 - LL0

    t = 0
    pred = np.array(list(map(lambda x: 1 if x >= t else 0, llRatio)))
    
    print("Binary tasks: MVG ML")
    acc, err = validate(pred, LTE)
    print("\tError rate(MVG ML): ", np.round(err*100, 1), "%")

    #################Tied####################
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    withinC = withinCovarianceMatrix([DC0, DC1])
    LL0, _ = loglikelihood(DTE, mu0, withinC)
    LL1, _ = loglikelihood(DTE, mu1, withinC)

    llRatio = LL1 - LL0

    t = 0
    pred = np.array(list(map(lambda x: 1 if x >= t else 0, llRatio)))
    
    print("Binary tasks: tied model")
    acc, err = validate(pred, LTE)
    print("\tError rate(tied): ", np.round(err*100, 1), "%")

    #################LDA#####################
    _, mu = datasetMean(DTR)
    Sb = betweenCovarianceMatrix([DTR[:, LTR == 0], DTR[:, LTR == 1]], mu)
    Sw = withinCovarianceMatrix([DTR[:, LTR == 0], DTR[:, LTR == 1]])

    W, _, _ = generalizedEigvalLDA(Sb, Sw, 1, DTR)

    DTR_lda = np.dot( W.T, DTR )
    DTE_lda = np.dot( W.T, DTE )

    predVal = LDAClassificator(DTR_lda, DTE_lda, LTR, LTE)

    print("LDA Classificator")
    acc, err = validate(predVal, LTE)
    print("\tError rate(LDA): ", np.round(err*100, 1), "%")

    #####Naive Bayes Gaussian Classifier######
    DC0, mu0 = datasetMean(DTR[:, LTR == 0])
    C0 = np.diag(np.diag(covarianceMatrix(DC0, DC0.shape[1])))
    LL0, _ = loglikelihood(DTE, mu0, C0)

    DC1, mu1 = datasetMean(DTR[:, LTR == 1])
    C1 = np.diag(np.diag(covarianceMatrix(DC1, DC1.shape[1])))
    LL1, _ = loglikelihood(DTE, mu1, C1)

    llRatio = LL1 - LL0

    t = 0
    pred = np.array(list(map(lambda x: 1 if x >= t else 0, llRatio)))
    
    print("Binary tasks: Naive Bayes Gaussian Classifier")
    acc, err = validate(pred, LTE)
    print("\tError rate(Naive Bayes): ", np.round(err*100, 1), "%")

def correlation(DTR, LTR, DTE, LTE):
    DC0, _ = datasetMean(DTR[:, LTR == 0])
    C0 = covarianceMatrix(DC0, DC0.shape[1])
    DC1, _ = datasetMean(DTR[:, LTR == 1])
    C1 = covarianceMatrix(DC1, DC1.shape[1])

    print("Covariance matrix class 0")
    print(C0)
    print("Covariance matrix class 1")
    print(C1)

    Corr0 = C0 / ( mcol(C0.diagonal()**0.5) * vrow(C0.diagonal()**0.5) )
    Corr1 = C1 / ( mcol(C1.diagonal()**0.5) * vrow(C1.diagonal()**0.5) )

    print("Correlation matrix class 0")
    print(np.matrix.round(Corr0, 5))
    print("Correlation matrix class 1")
    print(np.matrix.round(Corr1, 5))

def gaussianAssupmtion(DTR, LTR):
    for cls in range(2):
        print("Class: ", cls)

        for feature in range(6):
            classD = vrow(DTR[feature, LTR == cls])
            classDC, class_m_ML = datasetMean(classD)
            class_C_ML = covarianceMatrix(classDC, classD.shape[1])

            plt.figure()
            plt.hist(classD.ravel(), bins=50, density=True, label="class_"+str(cls)+"_feature_"+str(feature))
            XPlot = np.linspace(-8, 12, 1000)
            plt.legend()
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), class_m_ML, class_C_ML)))
            plt.savefig("lab05\\plots\\hists\\class_"+str(cls)+"_feature_"+str(feature)+".png")

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

    classification(DTR, LTR, DTE, LTE, 1, 6)

    correlation(DTR, LTR, DTE, LTE)

    #gaussianAssupmtion(DTR, LTR)

    classification(DTR, LTR, DTE, LTE, 1, 4)

    classification(DTR, LTR, DTE, LTE, 1, 2)
    classification(DTR, LTR, DTE, LTE, 3, 4)

    print("\n\nApplied PCA (m=5)")
    DtrC, mu = datasetMean(DTR)
    Ctr = covarianceMatrix(DtrC, DTR.shape[1])
    _, P = PCA(Ctr, 5, DTR)
        
    DTR_pca = np.dot( P.T, DTR )
    DTE_pca = np.dot( P.T, DTE )
    classification(DTR_pca, LTR, DTE_pca, LTE, 1, 6)