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
    i=0

    line = f.readline()
    while(line != ''):
        i+=1
        fields = line.split(',')
        x = np.array([float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])])
        x = mcol(x)
        
        l = -1
        match fields[4].replace('\n', ''):
            case 'Iris-virginica':
                l = 2
            case 'Iris-versicolor':
                l = 1
            case 'Iris-setosa':        
                l = 0

        datasetList.append(x)
        labelList.append(l)
        line = f.readline()

    print("Dataset loaded.")
    return np.hstack(datasetList), np.array(labelList)

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(DC,NSamples):
    C = ((DC) @ (DC).T) / float(NSamples)
    return C

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

def plotHistogram(D, L, direction):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #setosa
    D0 = D[:, L == 0]
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

    plt.hist(D0[direction,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[direction,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[direction,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel(direction)
    plt.legend()
    plt.show()

def plotScatters(D, L, directions = None):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #setosa
    D0 = D[:, L == 0]
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

    if (directions is None):
        rows, _ = D.shape
        for i in range(rows):
            for j in range(rows):
                if(i != j):
                    plt.figure()
                    plt.scatter(D0[i, :], D0[j, :], label="Setosa")
                    plt.scatter(D1[i, :], D1[j, :], label="Versicolor")
                    plt.scatter(D2[i, :], D2[j, :], label="Virginica")
                    plt.legend()
            plt.show()
    else:
        i = directions[0]
        j = directions[1]
        plt.scatter(D0[i, :], D0[j, :], label="Setosa")
        plt.scatter(D1[i, :], D1[j, :], label="Versicolor")
        plt.scatter(D2[i, :], D2[j, :], label="Virginica")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    fileName = "datasets/iris.csv"
    #D[0]-> sepal lenght
    #D[1]-> sepal width
    #D[2]-> petal lenght
    #D[3]-> petal width
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
    print("PCA calculation...")
    print("PCA m = 4")
    pca4, U4 = PCA(C, 4, D)
    print("Mine eighen vectors: ")
    print(U4)
    solution = np.load("lab03\\results\\IRIS_PCA_matrix_m4.npy")
    print("Solution eighen vectors:")
    print(solution)
    plotScatters(pca4, L, [0,1])
    plotHistogram(pca4, L, 0)
    print()

    #################################LDA####################################
    
    print("LDA calculation...")
    Sb = betweenCovarianceMatrix([D[:, L == 0], D[:, L == 1], D[:, L == 2]], mu)
    print("Between class covariance matrix")
    print(Sb)
    Sw = withinCovarianceMatrix([D[:, L == 0], D[:, L == 1], D[:, L == 2]])
    print("Within class covariance matrix:")
    print(Sw)

    print("My generalized eigvalue problem LDA matrix(m=2):")
    W, _, ldaGen2 = generalizedEigvalLDA(Sb, Sw, 2, D)
    print(W)
    print("My joint diagonalization problem LDA matrix(m=2):")
    W, _, ldaDiag2 = generalizedEigvalLDA(Sb, Sw, 2, D)
    print(W)
    solution = np.load("lab03\\results\\IRIS_LDA_matrix_m2.npy")
    print("Solution LDA matrix(m=2):")
    print(solution)
    
    plotScatters(ldaDiag2, L, [0,1])
    plotHistogram(ldaDiag2, L, 0)
