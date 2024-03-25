import numpy as np
import matplotlib.pyplot as plt

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

def plotHistograms(D, L):
    #fake
    D0 = D[:, L == 0]
    #genuine
    D1 = D[:, L == 1]

    i = 0
    rows, _ = D.shape
    for i in range(rows):
        plt.figure()
        #feature saving
        plt.hist(D0[i,:], density=True, alpha=0.5, label="Fake")
        plt.hist(D1[i,:], density=True, alpha=0.5, label="Genuine")
        plt.xlabel("Feature %d" %i)
        plt.legend()
        plt.savefig("lab02\\plots\\hists\\hist_feature%d.pdf" %i)
    plt.show()

def plotScatters(D, L):
    #fake
    D0 = D[:, L == 0]
    #genuine
    D1 = D[:, L == 1]

    rows, _ = D.shape
    for i in range(rows):
        for j in range(rows):
            if(i != j):
                plt.figure()
                plt.scatter(D0[i, :], D0[j, :], label="Fake")
                plt.scatter(D1[i, :], D1[j, :], label="Genuine")
                plt.legend()
                plt.savefig("lab02\\plots\\scatters\\scatter_x%d_y%d.pdf" % (i, j))
        plt.show()

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(D, mu):
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return C

if __name__ == "__main__":
    fileName = "datasets/trainData.txt"

    D, L = load(fileName)
    #plotHistograms(D,L)
    #plotScatters(D,L)

    DC, mu = datasetMean(D)
    C = covarianceMatrix(D, mu)
    print(C)

    var = D.var(1)
    std = D.std(1)
    print("Var: ", var, "\nStd: ", std)

    for cls in range(2):
        print("Class: ", cls)
        classD = D[:, L == cls]
        classDC, classMu = datasetMean(classD)

        print("Mean: ", classMu)
        print("Covariance matrix: \n", classDC)

        var = classD.var(1)
        std = classD.std(1)
        print("Var: ", var, "\nStd: ", std)

    