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

    return np.hstack(datasetList), np.array(labelList)

def plotHistograms(D, L):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #setosa
    D0 = D[:, L == 0]
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

    #sepal lenght
    plt.hist(D0[0,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[0,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[0,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel("Sepal lenght")
    plt.legend()
    plt.figure()

    #sepal width
    plt.hist(D0[1,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[1,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[1,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel("Sepal widht")
    plt.legend()
    plt.figure()
    
    #petal lenght
    plt.hist(D0[2,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[2,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[2,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel("Petal lenght")
    plt.legend()
    plt.figure()

    #petal widht
    plt.hist(D0[3,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[3,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[3,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel("Petal widht")
    plt.legend()
    plt.show()

def plotScatters(D, L):
    #set te columns with iris setosa to true, others to false; select from the dataset only iris setosa
    #setosa
    D0 = D[:, L == 0]
    #versicolor
    D1 = D[:, L == 1]
    #virginica
    D2 = D[:, L == 2]

    for i in range(4):
        for j in range(4):
            if(i != j):
                plt.figure()
                plt.scatter(D0[i, :], D0[j, :], label="Setosa")
                plt.scatter(D1[i, :], D1[j, :], label="Versicolor")
                plt.scatter(D2[i, :], D2[j, :], label="Virginica")
                plt.legend()
        plt.show()

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(D, mu):
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return C

if __name__ == '__main__':
    fileName = "datasets/iris.csv"
    #D[0]-> sepal lenght
    #D[1]-> sepal width
    #D[2]-> petal lenght
    #D[3]-> petal width
    D, L = load(fileName)

    DC, mu = datasetMean(D)
    #plotHistograms(DC, L)
    #plotScatters(DC, L)
    C = covarianceMatrix(D, mu)
    print(C)

    var = D.var(1)
    std = D.std(1)
    print("Var: ", var, "\nStd: ", std)

    for cls in [0,1,2]:
        print("Class: ", cls)
        classD = D[:, L == cls]
        classDC, classMu = datasetMean(classD)

        print("Mean: ", classMu)
        print("Covariance matrix: \n", classDC)

        var = classD.var(1)
        std = classD.std(1)
        print("Var: ", var, "\nStd: ", std)
