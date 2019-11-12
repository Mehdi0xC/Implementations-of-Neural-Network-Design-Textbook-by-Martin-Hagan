import numpy as np
from copy import deepcopy as cp

def initWeights(inputSize, outputSize):
    W = np.random.randn(inputSize, outputSize) / np.sqrt(inputSize)
    b = np.zeros(outputSize)
    return W.astype(np.float32), b.astype(np.float32)

def relu(x):
    return x * (x > 0)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def errorRate(targets, predictions):
    return np.mean(targets != predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def getNormalizerBox(inputData):
    featureSize = len(inputData[0])
    normalizerBox = np.zeros((2,featureSize))
    for i in range(featureSize):
        normalizerBox[0,i] = np.min(inputData[:,i]) 
        normalizerBox[1,i] = np.max(inputData[:,i]) 
    return normalizerBox

def normalizer(inputData):
    outputData = cp(inputData)
    normalizerBox = getNormalizerBox(inputData)
    for i in range(len(inputData)):
        for j in range(len(inputData[0])):
           outputData[i][j] = (inputData[i][j] - normalizerBox[0,j])/(normalizerBox[1,j] - normalizerBox[0,j])
    return outputData 

def getData(dataAddress):
    Y = []
    X = []
    i=0
    for line in open(dataAddress):
        row = line.split(',')
        Y.append(int(row[0])-1)
        tempList = []
        for element in row[1:-1] :
            tempList.append(float(element))
        X.append(tempList)
    X = np.array(X)
    Y = np.array(Y)  
    return X, Y