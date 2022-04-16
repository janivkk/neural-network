'''
TESTING
'''
#   workshop task -> implement slide 30 with multiple weights

import math

class NetworkTest:
    def __init__(self, i, w, b, t):
        self.input = i
        self.weights = w
        self.bias = b
        self.target = t
        self.rate = 0.1
        self.netArr = []
        self.netArr2 = []
        self.sigmoidArr = []
        self.hiddenError = []
        self.deltaArr = [[], [], [], []]
        self.outputArr = []
        self.softmaxArr = []

    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    def forward(self):
        #   first
        for neuron_weights in self.weights[:2]:
            net = 0
            for tempInput, tempWeight in zip(self.input, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        print(self.netArr)

        #   activation function [sigmoid]
        for tempNet in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[tempNet]))
        self.sigmoidArr.extend(self.bias)
        print(self.sigmoidArr)

        #   second
        for neuron_weights in self.weights[2:]:
            net = 0
            for tempWeight, weight in zip(self.sigmoidArr, neuron_weights):
                net += tempWeight * weight
            self.netArr2.append(net)
        print(self.netArr2)

    def backward(self):
        for i in range(len(self.target)):
            self.hiddenError.append(self.target[i] - self.netArr2[i])

        tempA = self.sigmoidArr[0] * ( 1 - self.sigmoidArr[0]) * ((self.weights[2][0] * self.hiddenError[0]) + (self.weights[3][0] * self.hiddenError[1]))
        self.hiddenError.append(tempA)
        #print(f"temp: {self.hiddenError}")

        tempB = self.sigmoidArr[1] * ( 1 - self.sigmoidArr[1]) * ((self.weights[2][1] * self.hiddenError[0]) + (self.weights[3][1] * self.hiddenError[1]))
        self.hiddenError.append(tempB)
        #print(f"Hidden errors: {self.hiddenError}")    

        for i in range(len(self.input)):
            self.deltaArr[0].append(self.rate * self.hiddenError[2] * self.input[i])

        for i in range(len(self.input)):
            self.deltaArr[1].append(self.rate * self.hiddenError[3] * self.input[i])

        for i in range(len(self.sigmoidArr)):
            self.deltaArr[2].append(self.rate * self.hiddenError[0] * self.sigmoidArr[i])

        for i in range(len(self.sigmoidArr)):
            self.deltaArr[3].append(self.rate * self.hiddenError[1] * self.sigmoidArr[i])
        #print(self.deltaArr)

        #   update weights
        #   psuedo code: weights[i] + deltaArr[i]
        #   works
        self.weights = [[x + y for x, y in zip(subListA, subListB)] for subListA, subListB in zip(self.weights, self.deltaArr)]

        #print(self.weights)
        
        #for x in self.weights:
            #print(" ".join(map(str, x)))

        return self.weights

    def softmax(self, o):
        for i in range(len(o)):
            _sum = math.exp(o[i]) / (math.exp(o[i] + math.exp(o[i])))
            self.softmaxArr.append(_sum)
        print(self.softmaxArr)

    def training(self):
        for neuron_weights in self.weights:
            temp = 0
            for out, weight in zip(self.target, neuron_weights):
                temp += out * weight
            self.outputArr.append(temp)
        print(self.outputArr)

#   inputs
x = [0, 1, 1]
bias = [1]
#   weights [2d]
weights = [[0.5, -0.2, 0.5], [0.1, 0.2, 0.3], [0.7, 0.6, 0.2], [0.9, 0.8, 0.4]]
#   desired output
desired_output = [1, 0]

#   training

#   softmax function...?

#   output errors then hidden errors in layer 1

#forwardStep()
#backwardStep()
#forwardStep()

network = NetworkTest(x, weights, bias, desired_output)
network.forward()
network.backward()
network.forward()
#network.training()

"""
for i in range(1):
    network.backward()
    network.forward()
    #network.backward()
    network.training()
"""


