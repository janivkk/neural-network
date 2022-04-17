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
    def __init__(self, w, t, o):
        self.weights = w
        self.target = t
        self.output = o     #   second
        self.rate = 0.1     #   learning rate
        self.hiddenArr = []     #   hidden errors array
        self.netArr = []    #   first 
        self.sigmoidArr = []    #   second
        self.softmaxArr = []    #   softmax / probability distribution

    #   sigmoid function with return type
    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    #   Softmax Function -> Softmax(Output)
    def softmax(self, output):  #   net
        for index in range(len(output)):
            self.softmaxArr.append(math.exp(output[index]) / (math.exp(output[0]) + math.exp(output[1])))

        print(f"Probaility Distribution: {self.softmaxArr}")

        return self.softmaxArr        

    #   forward step propagation
    def forward(self, perceptron, bias):
        #   add list to list -> input + bias
        perceptron.extend(bias)

        #   show old weights
        print("\nOLD:")
        for index in self.weights:
            print(" ".join(map(str, index)))

        #   first
        for neuron_weights in self.weights[:3]:
            net = 0
            for tempInput, tempWeight in zip(perceptron, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        print(f"\n[A4, A5, A6]: {self.netArr}")

        #   second
        #   sigmoid function [activation function]
        for i in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[i]))
        print(f"SIGMOID [A4, A5, A6]: {self.sigmoidArr}")

        #   second
        for neuron_weights in self.weights[3:]:
            net = 0
            for sWeight, tempWeight in zip(self.sigmoidArr, neuron_weights):
                net += sWeight * tempWeight
            self.output.append(net)
        print(f"Output [A7, A8]: {self.output}")

        self.netArr.clear()

        return self.output

    #   backward step propagation & update weights
    def back(self, perceptron):
        deltaArr = [[], [], [], [], []]  #  delta arr

        #   output errors -> calculating errors
        for i in range(len(self.target)):
            self.hiddenArr.append(self.target[i] - self.output[i])
        print(f"Output errors: {self.hiddenArr}")

        self.output.clear()

        #   hidden errors
        self.hiddenArr.append(self.sigmoidArr[0] * (1 - self.sigmoidArr[0]) * ((self.weights[3][0] * self.hiddenArr[0]) + (self.weights[4][0] * self.hiddenArr[1])))

        self.hiddenArr.append(self.sigmoidArr[1] * (1 - self.sigmoidArr[1]) * ((self.weights[3][1] * self.hiddenArr[0]) + (self.weights[4][1] * self.hiddenArr[1])))

        self.hiddenArr.append(self.sigmoidArr[2] * (1 - self.sigmoidArr[2]) * ((self.weights[3][2] * self.hiddenArr[0]) + (self.weights[4][2] * self.hiddenArr[1])))
        print(f"Hidden errors: {self.hiddenArr}")

        #   delta weights [before update]
        for j in range(len(perceptron)):
            deltaArr[0].append(self.rate * self.hiddenArr[2] * perceptron[j])

        for k in range(len(perceptron)):
            deltaArr[1].append(self.rate * self.hiddenArr[3] * perceptron[k])

        for l in range(len(perceptron)):
            deltaArr[2].append(self.rate * self.hiddenArr[4] * perceptron[l])
            
        for m in range(len(self.sigmoidArr)):
            deltaArr[3].append(self.rate * self.hiddenArr[0] * self.sigmoidArr[m])

        for n in range(len(self.sigmoidArr)):
            deltaArr[4].append(self.rate * self.hiddenArr[1] * self.sigmoidArr[n])

        self.sigmoidArr.clear()
        self.hiddenArr.clear()

        #   update weights
        self.weights = [[x + y for x, y in zip(subLstA, subLstB)] for subLstA, subLstB in zip(self.weights, deltaArr)]

        deltaArr.clear()

        #   printing updated weights
        print("\nNEW:")
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

#   END
