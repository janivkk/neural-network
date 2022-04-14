""" LIBRARIES """
import math

""" 
    Artificial Neural Network

    Endeavoring the Backpropagation Algorithm
        -> Updating /weights/ through forward, backward steps
            -> Further calculating hidden errors in hidden layers

    Janusz Snieg
    (25085325) - University of Lincoln

"""

class Network:
    #   constructor [initialiser]
    def __init__(self):
        self.input = []
        self.weights = []
        self.bias = []
        self.net = 2

    #   sigmoid function with return type
    def sigmoid(self):
        return 1 / (1 + math.exp(-self.net))
        pass

    #   forward step propagation
    def forward(self):
        pass

    #   backward step propagation & update weights
    def back(self):
        pass

    "   Softmax Function -> Softmax(Output)"
    def softmax():
        pass

    def training():
        pass
