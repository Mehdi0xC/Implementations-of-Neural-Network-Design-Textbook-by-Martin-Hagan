# Simple Implementation of Hamming Neural Networks
# Developed by Mehdi0xC, Summer 2018
import numpy as np
import activationFunctions as F
class HammingNetwork(object):
    def __init__(self, prototypes):
        self.feedForwardLayer = self.FeedFowardLayer(W=prototypes)
        self.recurrentLayer = self.RecurrentLayer()

    def classify(self, obj):
        a1 = self.feedForwardLayer.propagate(obj=obj)
        recurrent_result = self.recurrentLayer.propagate(initial_a=a1)
        return F.compet(recurrent_result)

    class FeedFowardLayer(object):
        def __init__(self, W):
            self.Weights = W
            self.bias = np.array(self.Weights.shape[0]).repeat(self.Weights.shape[0], axis=0).reshape((self.Weights.shape[0], 1))     
            self.transfer_function = np.vectorize(F.lin, otypes=[np.float])
        def propagate(self, obj):
            return self.transfer_function(self.Weights.dot(obj) + self.bias)


    class RecurrentLayer(object):
        def __init__(self, W = None):
            if W is None:
                self.Weights = W
            else:
                self.Weights = None
            self.transfer_function = np.vectorize(F.posl, otypes=[np.float])

        def propagate(self, initial_a):
            if self.Weights is None:
                s = initial_a.shape[0]
                epsilon = 1 / (s - 1)
                epsilon *= -1
                self.Weights = np.ones((s, s))
                for i in range(s):
                    for j in range(s):
                        if i != j:
                            self.Weights[i][j] = epsilon
            a2 = self.transfer_function(self.Weights.dot(initial_a))

            while True:
                a3 = self.transfer_function(self.Weights.dot(a2))
                if a2.all() != a3.all():
                    a2 = a3
                else:
                    return a3
