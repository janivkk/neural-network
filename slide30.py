'''
TESTING
'''
#   workshop task -> implement slide 30 with multiple weights

import math

class NetworkTest:
    def __init__(self, w, t):
        self.weights = w
        self.target = t
        self.rate = 0.1
        self.netArr = []
        self.netArr2 = []
        self.sigmoidArr = []
        self.hiddenError = []
        self.outputArr = []
        self.softmaxArr = []

    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    def squarederror(self, target, output):
        for index in range(len(target)):
            pass

    def forward(self, perceptron, bias):
        #   first
        for neuron_weights in self.weights[:2]:
            net = 0
            for tempInput, tempWeight in zip(perceptron, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        #print(f"Net 3, 4: {self.netArr} \n")

        #   activation function [sigmoid]
        for tempNet in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[tempNet]))
        self.sigmoidArr.extend(bias)
        #print(f"Sigmoid 3, 4: {self.sigmoidArr} \n")

        #   second
        for neuron_weights in self.weights[2:]:
            net = 0
            for tempWeight, weight in zip(self.sigmoidArr, neuron_weights):
                net += tempWeight * weight
            self.netArr2.append(net)
        #print(f"Net 5, 6: {self.netArr2}\n")

        self.netArr.clear()

        return self.netArr2

    def backward(self, perceptron):
        deltaArr = [[], [], [], []]

        for i in range(len(self.netArr2)):
            self.hiddenError.append(self.target[i] - self.netArr2[i])

        self.netArr2.clear()

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
        #   psuedo code: weights[i] + deltaArr[i]
        #   works
        self.weights = [[x + y for x, y in zip(subListA, subListB)] for subListA, subListB in zip(self.weights, deltaArr)]

        deltaArr.clear()

        #print(self.weights)
        
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

    def softmax(self, o):
        for i in range(len(o)):
            _sum = math.exp(o[i]) / (math.exp(o[i] + math.exp(o[i])))
            self.softmaxArr.append(_sum)
        print(self.softmaxArr)

    def training(self):
        for neuron_weights in self.weights[2:]:
            temp = 0
            for out, weight in zip(self.target, neuron_weights):
                temp += out * weight
            self.outputArr.append(temp)
        print(self.outputArr)

#   inputs
inputs = [0, 1, 1]
inputsUnseen = [0.4, 0.7, 1]
#   bias
bias = [1]
#   weights [2d]
weights = [[0.5, -0.2, 0.5], [0.1, 0.2, 0.3], [0.7, 0.6, 0.2], [0.9, 0.8, 0.4]]
#   desired output
desired_output = [1, 0]

network = NetworkTest(weights, desired_output)

for i in range(10):
    print(f"Step: {i}")
    network.forward(inputs, bias)
    network.backward(inputs)
#network.training()



