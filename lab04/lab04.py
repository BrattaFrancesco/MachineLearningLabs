import numpy as np
import matplotlib.pyplot as plt

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
    XND = np.load('lab04/solutions/XND.npy')
    mu = np.load('lab04/solutions/muND.npy')
    C = np.load('lab04/solutions/CND.npy')
    pdfSol = np.load('lab04/solutions/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    #plt.show()

    DC, m_ML = datasetMean(XND)
    C_ML = covarianceMatrix(DC, XND.shape[1])

    ll = loglikelihood(XND, m_ML, C_ML)
    print(ll)

    X1D = np.load('lab04/solutions/X1D.npy')
    print(X1D)
    DC, m_ML = datasetMean(X1D)
    C_ML = covarianceMatrix(DC, X1D.shape[1])

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
    plt.show()