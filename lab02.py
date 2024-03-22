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
        match fields[4]:
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
    dataset, labels = load(fileName)
    
    plt.hist(dataset)
    plt.show()