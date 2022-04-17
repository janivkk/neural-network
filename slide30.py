'''
TESTING
'''
#   workshop task -> implement slide 30 with multiple weights

import math
import matplotlib

class NetworkTest:
    def __init__(self, w, t, o):
        self.weights = w
        self.target = t
        self.rate = 0.1
        self.output = o
        self.netArr = []
        self.sigmoidArr = []
        self.hiddenError = []
        #self.outputArr = []
        self.softmaxArr = []

    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    def softmax(self, output):  #   net
        for index in range(len(output)):
            self.softmaxArr.append(math.exp(output[index]) / (math.exp(output[0]) + math.exp(output[1])))

        print(f"Probaility Distribution: {self.softmaxArr}")

        return self.softmaxArr

    def forward(self, perceptron, bias):
        perceptron.extend(bias)

        #   first
        for neuron_weights in self.weights[:2]:
            net = 0
            for tempInput, tempWeight in zip(perceptron, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        #print(f"Net 4, 5: {self.netArr} \n")

        #   activation function [sigmoid]
        for tempNet in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[tempNet]))
        #self.sigmoidArr.extend(bias)
        #print(f"Sigmoid 4, 5: {self.sigmoidArr} \n")

        #   second
        for neuron_weights in self.weights[2:]:
            net = 0
            for tempWeight, weight in zip(self.sigmoidArr, neuron_weights):
                net += tempWeight * weight
            self.output.append(net)
        print(f"Net 6, 7: {self.output}\n")

        self.netArr.clear()

        return self.output

    def backward(self, perceptron):
        deltaArr = [[], [], [], []]

        for i in range(len(self.output)):
            self.hiddenError.append(self.target[i] - self.output[i])

        self.output.clear()

        tempA = self.sigmoidArr[0] * ( 1 - self.sigmoidArr[0]) * ((self.weights[2][0] * self.hiddenError[0]) + (self.weights[3][0] * self.hiddenError[1]))
        self.hiddenError.append(tempA)
        #print(f"Output errors: {self.hiddenError} \n")

        tempB = self.sigmoidArr[1] * ( 1 - self.sigmoidArr[1]) * ((self.weights[2][1] * self.hiddenError[0]) + (self.weights[3][1] * self.hiddenError[1]))
        self.hiddenError.append(tempB)
        #print(f"Hidden errors: {self.hiddenError} \n")   

        for i in range(len(perceptron)):
            deltaArr[0].append(self.rate * self.hiddenError[2] * perceptron[i])

        for i in range(len(perceptron)):
            deltaArr[1].append(self.rate * self.hiddenError[3] * perceptron[i])

        for i in range(len(self.sigmoidArr)):
            deltaArr[2].append(self.rate * self.hiddenError[0] * self.sigmoidArr[i])

        for i in range(len(self.sigmoidArr)):
            deltaArr[3].append(self.rate * self.hiddenError[1] * self.sigmoidArr[i])
        #print(f"Delta: {deltaArr} \n")

        self.sigmoidArr.clear()
        self.hiddenError.clear()

        #   update weights
        #   works
        self.weights = [[x + y for x, y in zip(subListA, subListB)] for subListA, subListB in zip(self.weights, deltaArr)]

        deltaArr.clear()
        
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

    def training(self):
        pass

#   inputs
inputs = [0, 1, 1]
inputsUnseen = [0.4, 0.7, 1]
#   bias
bias = [1]
#   weights [2d]
weights = [[0.5, -0.2, 0.5], [0.1, 0.2, 0.3], [0.7, 0.6, 0.2], [0.9, 0.8, 0.4]]
#   desired output
desired_output = [1, 0]
#   output
output = []

network = NetworkTest(weights, desired_output, output)

for j in range(100):
    print(f"Epoch: {j}")
    for i in range(100):
        print(f"Step: {i}")
        network.forward(inputs, bias)
        network.backward(inputs)
#network.training()

network.forward(inputsUnseen, bias) #   expected: 1.055, 0.0347 for 6 & 7 [DONE!]

network.softmax(output) #?


