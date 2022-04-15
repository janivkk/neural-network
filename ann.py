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
    def __init__(self, i, w, b):
        self.input = i
        self.weights = w
        self.bias = b
        self.netArr = []
        self.sigmoidArr = []

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
        print(f"nets: {self.netArr}")

        #   sigmoid function
        for i in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[i]))
        print(f"sigmoid: {self.sigmoidArr}")

    #   backward step propagation & update weights
    def back(self):
        pass

    #   Softmax Function -> Softmax(Output)
    def softmax():
        pass

    def training():
        pass
