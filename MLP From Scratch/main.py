import nn
from utils import getData, normalizer

settings = {
    "numberOfHiddenLayerNeurons" : 32,
    "learningRate" : 0.0001,
    "epochs" : 10000,
    "testTrainSeperatingFactor" : 0.35,
    "reg" : 0,
    "showFigure" : True,
    "activationFunction" : "relu"
}



X, Y = getData('wine.data')
X = normalizer(X)
classifier = nn.ANN(settings)
classifier.fit(X, Y)
