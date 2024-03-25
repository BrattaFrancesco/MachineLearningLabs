import numpy as np
import matplotlib.pyplot as plt

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

def covarianceMatrix(DC, mu):
    C = ((DC) @ (DC).T) / float(D.shape[1])
    return C

def PCA(C, m, D):
    #get eigenvalues and eigenvectors, sorted from the smaller to the largest
    _, U = np.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]

    P = U[:, 0:m]

    #computing SVD
    U, _, _ = np.linalg.svd(C)

    DP = np.dot(P.T, D)

    return DP, U

def plotScatters(D, L):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #setosa
    D0 = D[:, L == 0]
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

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

if __name__ == "__main__":
    fileName = "datasets/iris.csv"
    #D[0]-> sepal lenght
    #D[1]-> sepal width
    #D[2]-> petal lenght
    #D[3]-> petal width
    D, L = load(fileName)

    DC, mu = datasetMean(D)
    C = covarianceMatrix(DC, mu)

    print("Mean:")
    print(mu)
    print("Covariance matrix:")
    print(C)

    print("PCA m = 4")
    pca4, U4 = PCA(C, 4, D)
    print("Mine eighen vectors: ")
    print(U4)
    a = np.load("lab03\\results\\IRIS_PCA_matrix_m4.npy")
    print("Prof eighen vectors:")
    print(a) 
    
