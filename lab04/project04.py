import numpy as np
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

def vrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

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

def datasetMean(D):
    mu = mcol(D.mean(1))
    DC = D - mu
    return DC, mu

def covarianceMatrix(DC,NSamples):
    C = ((DC) @ (DC).T) / float(NSamples)
    return C

def loglikelihood(XND, m_ML, C_ML):
    Y = logpdf_GAU_ND(XND, m_ML, C_ML)
    return Y.sum()

if __name__ == "__main__":
    fileName = "datasets/trainData.csv"

    D, L = load(fileName)

    for cls in range(2):
        print("Class: ", cls)

        for feature in range(6):
            classD = vrow(D[feature, L == cls])
            classDC, class_m_ML = datasetMean(classD)
            class_C_ML = covarianceMatrix(classDC, classD.shape[1])

            ll = loglikelihood(classD, class_m_ML, class_C_ML)
            print(ll)

            plt.figure()
            plt.hist(classD.ravel(), bins=50, density=True)
            XPlot = np.linspace(-8, 12, 1000)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), class_m_ML, class_C_ML)))
            plt.title("Class: " + str(cls) + " feature: " + str(feature))
            plt.savefig("lab04\\plots\\%s_%d.png" %(cls, feature))
    plt.show()
            
