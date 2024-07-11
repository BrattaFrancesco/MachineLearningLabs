import numpy
import scipy.special
import bayesRisk
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load(fileName):
    print("Loading dataset...")
    f=open(fileName, 'r')
    
    datasetList = []
    labelList = []

    line = f.readline()
    while(line != ''):
        fields = line.split(',')
        x = numpy.array([float(i) for i in fields[0:-1]])
        x = vcol(x)
        l = int(fields[len(fields)-1].replace("\n", ""))
        datasetList.append(x)
        labelList.append(l)
        line = f.readline()

    return numpy.hstack(datasetList), numpy.array(labelList)

# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

def linearSVM(DTR, LTR, DVAL, LVAL):
    K = 1.0
    Cs = numpy.logspace(-5, 0, 11)
    minDCFs = []
    actDCFs = []
    for C in Cs:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.5: %.4f' % minDCF)
        minDCFs.append(minDCF)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('actDCF - pT = 0.5: %.4f' % actDCF)
        actDCFs.append(actDCF)
        print ()
    plt.plot(Cs, minDCFs, label='minDCF')
    plt.plot(Cs, actDCFs, label='actDCF')
    plt.xscale('log', base=10)
    plt.legend()
    plt.show()

def polyKernelSVM(DTR, LTR, DVAL, LVAL):
    kernelFunc = polyKernel(2, 1)
    eps = 0.0
    minDCFs = []
    actDCFs = []
    for C in [1.0, 2.0, 3.0, 4.0]:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))
        print("C: ", C)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.5: %.4f' % minDCF)
        minDCFs.append(minDCF)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('actDCF - pT = 0.5: %.4f' % actDCF)
        actDCFs.append(actDCF)
        print ()

def rbfKernelSVM(DTR, LTR, DVAL, LVAL):
    for gamma in [numpy.exp(-4),numpy.exp(-3),numpy.exp(-2),numpy.exp(-1),]:
        minDCFs = []
        actDCFs = []
        Cs = numpy.logspace(-3, 2, 11)
        for C in Cs:
            kernelFunc = rbfKernel(gamma)
            eps = 1.0
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print ('Error rate: %.1f' % (err*100))
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            print ('minDCF - pT = 0.5: %.4f' % minDCF)
            minDCFs.append(minDCF)
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            print ('actDCF - pT = 0.5: %.4f' % actDCF)
            actDCFs.append(actDCF)
            print ()
        plt.plot(Cs, minDCFs, label='minDCF gamma:' + str(numpy.round(gamma, 3)))
        plt.plot(Cs, actDCFs, label='actDCF gamma:' + str(numpy.round(gamma, 3)))
    plt.xscale('log', base=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.show()
        

if __name__ == '__main__':
    fileName = "datasets/trainData.csv"

    D, L = load(fileName)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    linearSVM(DTR, LTR, DVAL, LVAL)

    polyKernelSVM(DTR, LTR, DVAL, LVAL)
            
    rbfKernelSVM(DTR, LTR, DVAL, LVAL)
    