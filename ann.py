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
    def __init__(self, i, w, b, t):
        self.input = i
        self.weights = w
        self.bias = b
        self.target = t
        self.hiddenArr = []     #   hidden errors array
        self.netArr = []    #   first
        self.tempNetArr = []    #  second
        self.sigmoidArr = []    #   second

    #   sigmoid function with return type
    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    #   forward step propagation
    def forward(self):
        #   add list to list -> input + bias
        self.input.extend(self.bias)

        #   first
        for neuron_weights in self.weights[:3]:
            net = 0
            for tempInput, tempWeight in zip(self.input, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        print(f"nets 4, 5, 6: {self.netArr}")

        #   second
        #   sigmoid function [activation function]
        for i in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[i]))
        self.sigmoidArr.extend(self.bias)
        print(f"sigmoid 7, 8: {self.sigmoidArr}")

        #   second
        for neuron_weights in self.weights[3:]:
            net = 0
            for sWeight, tempWeight in zip(self.sigmoidArr, neuron_weights):
                net += sWeight * tempWeight
            self.tempNetArr.append(net)
        print(f"net 7, 8: {self.tempNetArr}")

        return self.tempNetArr

    #   backward step propagation & update weights
    def back(self):
        #   output errors -> calculating errors
        for i in range(len(self.target)):
            temp = self.target[i] - self.tempNetArr[i]
            self.hiddenArr.append(temp)
        print(f"Output errors {self.hiddenArr}")

        #   hidden errors

    #   Softmax Function -> Softmax(Output)
    def softmax():
        pass

    def training():
        pass
