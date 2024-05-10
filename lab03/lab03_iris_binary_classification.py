import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lalg

def load_iris(): 
    return datasets.load_iris()['data'].T, datasets.load_iris ()['target'] 

def mcol(v):
    return v.reshape((v.size, 1))

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

def jointDiagonalLDA(Sb, Sw, m, D):
    #computing P1= U*Σ−1/2*U.T
    #Where s contains thr diagonal of Σ, and the diagonal of Σ-1/2 = diag(1.0/(s**0.5))
    U, s, _ = np.linalg.svd(Sw)

    P1 = np.dot( np.dot( U, np.diag(1.0/(s**0.5))), U.T )

    #computing P2 as eigenvector of Sbt = P1*Sb*P1.T
    Sbt = np.dot( np.dot( P1, Sb ), P1.T )

    _, U = np.linalg.eigh(Sbt)
    P2 = U[:, :m]

    #find W matrix of direction of LDA
    W = np.dot( P1.T, P2 )

    #project the dataset on W to find the LDA
    LDA = np.dot( W.T, D )

    return W, LDA

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
    threshold = (DTR[0, LTR == 1].mean() + DTR[0, LTR == 2].mean()) / 2.0 # We only have one dimension in the projected Dataset
    
    PredVal = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PredVal[DVAL[0] >= threshold] = 2
    PredVal[DVAL[0] < threshold] = 1

    return PredVal

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

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def plotHistogram(D, L, direction):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

    plt.hist(D1[direction,:], bins=5, density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[direction,:], bins=5, density=True, alpha=0.5, label="Virginica")
    plt.xlabel(direction)
    plt.legend()
    plt.show()

if __name__ == "__main__":                                                            
    DIris, LIris = load_iris() 
    D = DIris[:, LIris != 0] 
    L = LIris[LIris != 0]

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    ########################LDA classificator##############################
    _, mu = datasetMean(DTR)
    Sb = betweenCovarianceMatrix([DTR[:, LTR == 1], DTR[:, LTR == 2]], mu)
    Sw = withinCovarianceMatrix([DTR[:, LTR == 1], DTR[:, LTR == 2]])

    W, _, _ = generalizedEigvalLDA(Sb, Sw, 1, DTR)

    DTR_lda = np.dot( W.T, DTR )
    DVAL_lda = np.dot( W.T, DVAL )

    #plotHistogram(DTR_lda, LTR, 0)
    #plotHistogram(DVAL_lda, LVAL, 0)
    predVal = LDAClassificator(DTR_lda, DVAL_lda, LTR, LVAL)

    print("Error rate: ", np.round(predVal[predVal != LVAL].shape[0]/predVal.shape[0]*100, 1))

    #########################PCA classificator#############################
    DtrC, mu = datasetMean(DTR)
    Ctr = covarianceMatrix(DtrC, DTR.shape[1])

    _, P = PCA(Ctr, 1, DTR)

    DTR_pca = np.dot( P.T, DTR )
    DVAL_pca = np.dot( P.T, DVAL )

    plotHistogram(DTR_pca, LTR, 0)
    plotHistogram(DVAL_pca, LVAL, 0)
    #we observe that the PCA is not that effective as the LDA

    #########################LDA + PCA classificator#######################
    DtrC, mu = datasetMean(DTR)
    Ctr = covarianceMatrix(DtrC, DTR.shape[1])

    _, P = PCA(Ctr, 2, DTR)
    
    DTR_pca = np.dot( P.T, DTR )
    DVAL_pca = np.dot( P.T, DVAL )

    _, mu = datasetMean(DTR_pca)
    Sb = betweenCovarianceMatrix([DTR_pca[:, LTR == 1], DTR_pca[:, LTR == 2]], mu)
    Sw = withinCovarianceMatrix([DTR_pca[:, LTR == 1], DTR_pca[:, LTR == 2]])

    W, _, _ = generalizedEigvalLDA(Sb, Sw, 1, DTR_pca)

    DTR_lda = np.dot( W.T, DTR_pca )
    DVAL_lda = np.dot( W.T, DVAL_pca )

    plotHistogram(DTR_lda, LTR, 0)
    plotHistogram(DVAL_lda, LVAL, 0)
    predVal = LDAClassificator(DTR_lda, DVAL_lda, LTR, LVAL)

    print(predVal[predVal != LVAL].shape[0]/predVal.shape[0])