import numpy as np
import matplotlib.pyplot as plt

from utils import getData, softmax, cost2, y2indicator, errorRate, relu
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self,settings):
        self.numberOfHiddenLayerNeurons = settings["numberOfHiddenLayerNeurons"]
        self.learningRate = settings["learningRate"]
        self.reg = settings["reg"]
        self.epochs = settings["epochs"]
        self.showFigure = settings["showFigure"]
        self.testTrainSeperatingFactor = settings["testTrainSeperatingFactor"]
        self.activationFunction = settings["activationFunction"]

    def fit(self, X, Y):
        X, Y = shuffle(X, Y)
        totalSampleCount, _ = X.shape
        testTrainSeperationIndex = int(self.testTrainSeperatingFactor*totalSampleCount)
        Xvalid, Yvalid = X[-testTrainSeperationIndex:], Y[-testTrainSeperationIndex:]
        X, Y = X[:-testTrainSeperationIndex], Y[:-testTrainSeperationIndex]

        numberOfSamples , featureVectorSize = X.shape
        classesCount = len(set(Y))
        target = y2indicator(Y)
        #input to hidden layer weights and biases
        self.W1 = np.random.randn(featureVectorSize, self.numberOfHiddenLayerNeurons) / np.sqrt(featureVectorSize + self.numberOfHiddenLayerNeurons)
        self.b1 = np.zeros(self.numberOfHiddenLayerNeurons)
        #hidden layer to output weights and biases
        self.W2 = np.random.randn(self.numberOfHiddenLayerNeurons, classesCount) / np.sqrt(self.numberOfHiddenLayerNeurons + classesCount)
        self.b2 = np.zeros(classesCount)

        costs = []
        bestValidationError = 1
        for i in range(self.epochs):
            # forward propagation and cost calculation
            output, hiddenLayerOutput = self.forward(X)

            # gradient descent step
            distance = target - output




            self.W2 += self.learningRate*(hiddenLayerOutput.T.dot(distance) + self.reg*self.W2)
            self.b2 += self.learningRate*(distance.sum(axis=0) + self.reg*self.b2)
            dOutput = distance.dot(self.W2.T) * (hiddenLayerOutput > 0) # relu
            self.W1 += self.learningRate*(X.T.dot(dOutput) + self.reg*self.W1)
            self.b1 += self.learningRate*(dOutput.sum(axis=0) + self.reg*self.b1)

            if i % 10 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = errorRate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < bestValidationError:
                    bestValidationError = e
        print("bestValidationError:", bestValidationError)

        if self.showFigure:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        if(self.activationFunction == "relu"):
            hiddenLayerOutput = relu(X.dot(self.W1) + self.b1)
        else:   
            hiddenLayerOutput = tanh(X.dot(self.W1) + self.b1)            
        return softmax(hiddenLayerOutput.dot(self.W2) + self.b2) , hiddenLayerOutput

    def predict(self, X):
        output, _ = self.forward(X)
        return np.argmax(output, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - errorRate(Y, prediction)
