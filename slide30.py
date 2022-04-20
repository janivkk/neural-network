'''
TESTING
'''
#   workshop task -> implement slide 30 with multiple weights

import math
import matplotlib.pyplot as plt

class NetworkTest:
    def __init__(self, w, t, o):
        self.weights = w
        self.target = t
        self.rate = 0.1
        self.output = o
        self.netArr = []
        self.sigmoidArr = []
        self.hiddenErr = []
        self.softmaxArr = []
        self.outputErr = [] #   output errors
        self.meanArr = []
        self.logged_error = []

    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    def softmax(self, output):  #   net
        for index in range(len(output)):
            self.softmaxArr.append(math.exp(output[index]) / (math.exp(output[0]) + math.exp(output[1])))

        print(f"Probaility Distribution: {self.softmaxArr}")

        return self.softmaxArr

    def forward(self, perceptron, bias):
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
        self.sigmoidArr.extend(bias)
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

    def meanErr(self):
        for i in range(len(self.output)):
            temp = (self.target[i]**2) - (self.output[i]**2)
            sqrTemp = pow(temp, 2)
            self.outputErr(sqrTemp)

        #self.output.clear()

        temp = 0
        for j in range(len(self.outputErr)):
            temp += self.outputErr[j] / 2
        self.meanArr.append(temp)

    def backward(self, perceptron):
        deltaArr = [[], [], [], []]

        #   6, 7
        for i in range(len(self.output)):
            self.hiddenErr.append(self.target[i] - self.output[i])
            #self.outputErr.append(((self.target[i] - self.output[i]) ** 2) / 2)
        #print(self.outputErr)

        self.output.clear()

        tempA = self.sigmoidArr[0] * ( 1 - self.sigmoidArr[0]) * ((self.weights[2][0] * self.hiddenErr[0]) + (self.weights[3][0] * self.hiddenErr[1]))
        self.hiddenErr.append(tempA)
        #print(f"Output errors: {self.hiddenErr} \n")

        tempB = self.sigmoidArr[1] * ( 1 - self.sigmoidArr[1]) * ((self.weights[2][1] * self.hiddenErr[0]) + (self.weights[3][1] * self.hiddenErr[1]))
        self.hiddenErr.append(tempB)
        #print(f"Hidden errors: {self.hiddenErr} \n")   

        for i in range(len(perceptron)):
            deltaArr[0].append(self.rate * self.hiddenErr[2] * perceptron[i])

        for i in range(len(perceptron)):
            deltaArr[1].append(self.rate * self.hiddenErr[3] * perceptron[i])

        for i in range(len(self.sigmoidArr)):
            deltaArr[2].append(self.rate * self.hiddenErr[0] * self.sigmoidArr[i])

        for i in range(len(self.sigmoidArr)):
            deltaArr[3].append(self.rate * self.hiddenErr[1] * self.sigmoidArr[i])
        #print(f"Delta: {deltaArr} \n")

        sqrArr.append(self.hiddenErr[0])
        sqrArr.append(self.hiddenErr[1])

        self.sigmoidArr.clear()
        self.hiddenErr.clear()

        #   update weights
        #   works
        self.weights = [[x + y for x, y in zip(subListA, subListB)] for subListA, subListB in zip(self.weights, deltaArr)]

        deltaArr.clear()
        
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

    def plotLearningCurve(self, strIn):
        x_data = []
        y_data = []
        x_data.extend([self.logged_error[i][0] for i in range(0, len(self.logged_error))])
        y_data.extend([self.logged_error[i][1] for i in range(0, len(self.logged_error))])
        fig, ax = plt.subplots()
        fig.suptitle(strIn)
        ax.set(xlabel = 'Epoch', ylabel = 'Squared Error')
        ax.plot(x_data, y_data, 'tab:green')
        plt.show()

    def training(self, perceptron, output, unseen, bias):
        epoch = int(input("Amount of Epochs: "))
        step = int(input("Amount of Steps: "))

        for i in range(epoch):
            print(f"Epoch :: {i} || Dataset :: {perceptron}")
            for j in range(step):
                print(f"Step :: {j}")
                self.forward(perceptron, bias)
                self.backward(perceptron)
            self.meanErr()

            self.logged_error.append([i, self.meanArr])
        
        self.plotLearningCurve("Mean Squarred Error per Epoch")

        self.forward(unseen, bias)

        self.softmax(output)

#   inputs
inputs = [0, 1, 1]
unseen = [0.4, 0.7, 1]
#   bias
bias = [1]
#   weights [2d]
weights = [[0.5, -0.2, 0.5], [0.1, 0.2, 0.3], [0.7, 0.6, 0.2], [0.9, 0.8, 0.4]]
#   desired output
desired_output = [1, 0]
#   output
output = []

sqrArr = []

network = NetworkTest(weights, desired_output, output)

""" for j in range(100):
    print(f"Epoch: {j}")
    for i in range(100):
        print(f"Step: {i}")
        network.forward(inputs, bias)
        network.alloutErr(desired_output)
        network.backward(inputs, sqrArr)

    network.caltErr()

#network.plotLearningCurve(sqrArr)

network.forward(inputsUnseen, bias) #   expected: 1.055, 0.0347 for 6 & 7 [DONE!]

network.softmax(output) #   expected: 0.735 & 0.265 """

network.training(inputs, output, unseen, bias)



