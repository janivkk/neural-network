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
        self.rate = 0.1 #   learning rate
        self.hiddenArr = []     #   hidden errors array
        self.netArr = []    #   first
        self.tempNetArr = []    #  second
        self.sigmoidArr = []    #   second
        self.deltaArr = [[], [], [], [], []]  #   delta
        self.updatedArr = [[], [], [], [], []]    #   updated weights Arr

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
        #print(f"nets 4, 5, 6: {self.netArr}")

        #   second
        #   sigmoid function [activation function]
        for i in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[i]))
        self.sigmoidArr.extend(self.bias)
        #print(f"sigmoid 4, 5, 6 + bias: {self.sigmoidArr}")

        #   second
        for neuron_weights in self.weights[3:]:
            net = 0
            for sWeight, tempWeight in zip(self.sigmoidArr, neuron_weights):
                net += sWeight * tempWeight
            self.tempNetArr.append(net)
        #print(f"net 7, 8: {self.tempNetArr}")

        return self.tempNetArr

    #   backward step propagation & update weights
    def back(self):
        #   output errors -> calculating errors
        for i in range(len(self.target)):
            temp = self.target[i] - self.tempNetArr[i]
            self.hiddenArr.append(temp)
        #print(f"Output errors: {self.hiddenArr}")

        #   hidden errors
        tempA = self.sigmoidArr[0] * (1 - self.sigmoidArr[0]) * ((self.weights[3][0] * self.hiddenArr[0]) + (self.weights[4][0] * self.hiddenArr[1]))
        self.hiddenArr.append(tempA)

        tempB = self.sigmoidArr[1] * (1 - self.sigmoidArr[1]) * ((self.weights[3][1] * self.hiddenArr[0]) + (self.weights[4][1] * self.hiddenArr[1]))
        self.hiddenArr.append(tempB)

        tempC = self.sigmoidArr[2] * (1 - self.sigmoidArr[2]) * ((self.weights[3][2] * self.hiddenArr[0]) + (self.weights[4][2] * self.hiddenArr[1]))
        self.hiddenArr.append(tempC)
        #print(f"Hidden errors: {self.hiddenArr}")

        #   delta weights [before update]
        for j in range(len(self.input)):
            self.deltaArr[0].append(self.rate * self.hiddenArr[2] * self.input[j])

        for k in range(len(self.input)):
            self.deltaArr[1].append(self.rate * self.hiddenArr[3] * self.input[k])

        for l in range(len(self.input)):
            self.deltaArr[2].append(self.rate * self.hiddenArr[4] * self.input[l])
            
        for m in range(len(self.sigmoidArr)):
            self.deltaArr[3].append(self.rate * self.hiddenArr[0] * self.sigmoidArr[m])

        for n in range(len(self.sigmoidArr)):
            self.deltaArr[4].append(self.rate * self.hiddenArr[1] * self.sigmoidArr[n])

        #   update weights
        self.weights = [[x + y for x, y in zip(subLstA, subLstB)] for subLstA, subLstB in zip(self.weights, self.deltaArr)]

        #   printing weights
        print("Delta")
        for x in self.deltaArr:
            print(" ".join(map(str, x)))

        print("Updated")
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

    #   Softmax Function -> Softmax(Output)
    def softmax():
        pass

    def training():
        pass
