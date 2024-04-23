import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lalg

def mcol(v):
    return v.reshape((v.size, 1))

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
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def plotHistograms(D, L, direction, prefix_filename):
    #fake
    D0 = D[:, L == 0]
    #genuine
    D1 = D[:, L == 1]

    plt.hist(D0[direction,:], density=True, alpha=0.5, label="Fake")
    plt.hist(D1[direction,:], density=True, alpha=0.5, label="Genuine")
    plt.xlabel("Feature %d" %direction)
    plt.legend()
    plt.savefig("lab03\\plots\\hists\\%s%d.png" %(prefix_filename, direction))
    plt.figure()

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(D, mu):
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
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
    threshold = (DTR[0, LTR == 0].mean() + DTR[0, LTR == 1].mean()) / 2.0 # We only have one dimension in the projected Dataset
    
    PredVal = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PredVal[DVAL[0] >= threshold] = 1
    PredVal[DVAL[0] < threshold] = 0

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

if __name__ == "__main__":
    fileName = "datasets/trainData.csv"

    D, L = load(fileName)
    
    #standardize the dataset arount the mean
    DC, mu = datasetMean(D)
    C = covarianceMatrix(DC, D.shape[1])

    print("Mean(µ):")
    print(mu)
    print("Covariance matrix(C):")
    print(C)
    print()

    #################################PCA####################################
    
    #calculate the PCA with new dimensionality, in this case always 4
    print("Task 1\nPCA calculation...")
    print("PCA m = 6")
    pca, _= PCA(C, 6, D)
    
    for i in range(6):
        plotHistograms(pca, L, i, "pca6_feature_")

    print()

    #################################LDA####################################
    
    print("Task 2\nLDA calculation...")
    Sb = betweenCovarianceMatrix([D[:, L == 0], D[:, L == 1]], mu)
    print("Between class covariance matrix")
    print(Sb)
    Sw = withinCovarianceMatrix([D[:, L == 0], D[:, L == 1]])
    print("Within class covariance matrix:")
    print(Sw)

    print("My generalized eigvalue problem LDA matrix(m=1):")
    W, _, ldaGen = generalizedEigvalLDA(Sb, Sw, 1, D)
    print(W)
    print("My joint diagonalization problem LDA matrix(m=1):")
    W, ldaDiag = jointDiagonalLDA(Sb, Sw, 1, D)
    print(W)
    
    ldaGen = -ldaGen
    ldaDiag = -ldaDiag
    plotHistograms(ldaGen, L, 0, "lda_gen_dimensionality_")
    plotHistograms(ldaDiag, L, 0, "lda_diag_dimensionality_")
    
    ########################LDA classificator##############################
    print("Task 3\nLDA classificator")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    _, mu = datasetMean(DTR)
    Sb = betweenCovarianceMatrix([DTR[:, LTR == 0], DTR[:, LTR == 1]], mu)
    Sw = withinCovarianceMatrix([DTR[:, LTR == 0], DTR[:, LTR == 1]])

    W, _, _ = generalizedEigvalLDA(Sb, Sw, 1, DTR)

    DTR_lda = np.dot( W.T, DTR )
    DVAL_lda = np.dot( W.T, DVAL )

    plotHistograms(DTR_lda, LTR, 0, "DTR_lda_dimensionality_")
    plotHistograms(DVAL_lda, LVAL, 0, "DVAL_lda_dimensionality_")
    predVal = LDAClassificator(DTR_lda, DVAL_lda, LTR, LVAL)

    print("Predicted:\t", predVal)
    print("Actual:\t", LVAL)

    print("Error rate: ", predVal[predVal != LVAL].shape[0]/predVal.shape[0])

    #########################LDA + PCA classificator#######################
    print("Task 5\n LDA + PCA")
    DtrC, mu = datasetMean(DTR)
    Ctr = covarianceMatrix(DtrC, DTR.shape[1])

    for m in [2,3,4,5]:
        _, P = PCA(Ctr, m, DTR)
        
        DTR_pca = np.dot( P.T, DTR )
        DVAL_pca = np.dot( P.T, DVAL )

        _, mu = datasetMean(DTR_pca)
        Sb = betweenCovarianceMatrix([DTR_pca[:, LTR == 0], DTR_pca[:, LTR == 1]], mu)
        Sw = withinCovarianceMatrix([DTR_pca[:, LTR == 0], DTR_pca[:, LTR == 1]])

        W, _, _ = generalizedEigvalLDA(Sb, Sw, 1, DTR_pca)

        DTR_lda = np.dot( W.T, DTR_pca )
        DVAL_lda = np.dot( W.T, DVAL_pca )

        plotHistograms(DTR_lda, LTR, 0, f"DTR_lda_prepca_m={m}_dimensionality_")
        plotHistograms(DVAL_lda, LVAL, 0, f"DVAL_lda_prepca_m={m}_dimensionality_")
        predVal = LDAClassificator(DTR_lda, DVAL_lda, LTR, LVAL)

        print(f"Error rate (m={m}): ", predVal[predVal != LVAL].shape[0]/predVal.shape[0])
