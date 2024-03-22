import numpy as np
import matplotlib.pyplot as plt

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
        x = x.reshape(x.size, 1)
        
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
        
if __name__ == '__main__':
    fileName = "dataset/iris.csv"
    #D[0]-> sepal lenght
    #D[1]-> sepal width
    #D[2]-> petal lenght
    #D[3]-> petal width
    D, L = load(fileName)

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
    plt.hist(D0[2,:], density=True, alpha=0.5, label="Setosa")
    plt.hist(D1[2,:], density=True, alpha=0.5, label="Versicolor")
    plt.hist(D2[2,:], density=True, alpha=0.5, label="Virginica")
    plt.xlabel("Petal widht")
    plt.legend()
    plt.show()