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
    #   initialiser [constructor]
    def __init__(self, inputs, weights):
        self.input = inputs
        self.weights = weights
        self.bias = []
        self.netArr = []

    #   sigmoid function with return type
    def sigmoid(self):
        return 1 / (1 + math.exp(-self.net))

    #   forward step propagation
    def forward(self):
        #   first
        for neuron_weights in self.weights[:2]:
            net = 0
            for tempInput, tempWeight in zip(self.input, neuron_weights):
                net += tempInput * tempWeight
                self.netArr.append(net)
        print(self.netArr)

    #   backward step propagation & update weights
    def back(self):
        pass

    #   Softmax Function -> Softmax(Output)
    def softmax():
        pass

    def training():
        pass
