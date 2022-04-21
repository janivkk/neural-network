'''
//  Backpropagation algorithm

'''
" LIBRARIES "
#import mlp_network

" CLASS "
" LIBRARIES "
import math

import matplotlib.pyplot as plt

""" 
    Artificial Neural Network

    Endeavoring the Backpropagation Algorithm
        -> Updating /weights/ through forward, backward steps
            -> Further calculating hidden errors in hidden layers

    Janusz Snieg
    (25085325) - University of Lincoln

"""

class MLP_Network:
    #   initialiser [constructor]
    def __init__(self, w, o):
        self.weights = w
        self.output = o     #   second
        self.lr = 0.1     #   learning rate
        self.hiddenArr = []     #   hidden errors array
        self.netArr = []    #   first 
        self.sigmoidArr = []    #   second
        self.softmaxArr = []    #   softmax / probability distribution
        self.errArr = []   #   sqr error array
        self.logged_error = []
        self.total = 0

    #   sigmoid function with return type
    def sigmoid(self, n):
        return 1 / (1 + math.exp(-n))

    #   Softmax Function -> Softmax(Output)
    def softmax(self, output):
        for index in range(len(output)):
            self.softmaxArr.append(math.exp(output[index]) / (math.exp(output[0]) + math.exp(output[1])))

        print(f"Probaility Distribution: {self.softmaxArr}")

        return self.softmaxArr    

    #   forward step propagation
    def forward(self, perceptron, bias):
        #   add list to list -> input + bias
        perceptron.extend(bias)

        print(f"\nForward Step: Input + Bias => {perceptron}\n")

        #   show old weights
        print("\nOLD [BIAS 'END']:")
        for index in self.weights:
            print(" ".join(map(str, index)))

        #   first
        for neuron_weights in self.weights[:3]:
            net = 0
            for tempInput, tempWeight in zip(perceptron, neuron_weights):
                net += tempInput * tempWeight
            self.netArr.append(net)
        print(f"\n[A4, A5, A6]: {self.netArr}")

        #   second => sigmoid function [activation function]
        for i in range(len(self.netArr)):
            self.sigmoidArr.append(self.sigmoid(self.netArr[i]))
        self.sigmoidArr.extend(bias)
        print(f"SIGMOID => [A4, A5, A6]: {self.sigmoidArr}")

        #   second
        for neuron_weights in self.weights[3:]:
            net = 0
            for sWeight, tempWeight in zip(self.sigmoidArr, neuron_weights):
                net += sWeight * tempWeight
            self.output.append(net)
        print(f"[A7, A8]: {self.output}")

        self.netArr.clear()

        return self.output

    #   calculate mean square error
    def getErr(self):
        lst = [i ** 2 for i in self.errArr]   #   lst = []
        y = sum(lst)
        self.total = y / len(self.errArr)
        return self.total

    #   backward step propagation & update weights
    def back(self, perceptron, target):
        deltaArr = [[], [], [], [], []]  #  delta arr

        print(f"\nBackward Step: Input + Bias => {perceptron}\n")

        #   output errors -> calculating errors
        for i in range(len(self.output)):
            self.hiddenArr.append(target[i] - self.output[i])
            #arr.append(target[i] - self.output[i])
            self.errArr.append(target[i] - self.output[i])
        print(f"OUTPUT ERRORS: {self.hiddenArr}")

        self.output.clear()

        #   hidden errors
        self.hiddenArr.append(self.sigmoidArr[0] * (1 - self.sigmoidArr[0]) * ((self.weights[3][0] * self.hiddenArr[0]) + (self.weights[4][0] * self.hiddenArr[1])))

        self.hiddenArr.append(self.sigmoidArr[1] * (1 - self.sigmoidArr[1]) * ((self.weights[3][1] * self.hiddenArr[0]) + (self.weights[4][1] * self.hiddenArr[1])))

        self.hiddenArr.append(self.sigmoidArr[2] * (1 - self.sigmoidArr[2]) * ((self.weights[3][2] * self.hiddenArr[0]) + (self.weights[4][2] * self.hiddenArr[1])))
        print(f"HIDDEN ERRORS: {self.hiddenArr}")

        #   delta weights [before update]
        for j in range(len(perceptron)):
            deltaArr[0].append(self.lr * self.hiddenArr[2] * perceptron[j])

        for k in range(len(perceptron)):
            deltaArr[1].append(self.lr * self.hiddenArr[3] * perceptron[k])

        for l in range(len(perceptron)):
            deltaArr[2].append(self.lr * self.hiddenArr[4] * perceptron[l])
            
        for m in range(len(self.sigmoidArr)):
            deltaArr[3].append(self.lr * self.hiddenArr[0] * self.sigmoidArr[m])

        for n in range(len(self.sigmoidArr)):
            deltaArr[4].append(self.lr * self.hiddenArr[1] * self.sigmoidArr[n])

        self.sigmoidArr.clear()
        self.hiddenArr.clear()

        #   update weights
        self.weights = [[x + y for x, y in zip(subLstA, subLstB)] for subLstA, subLstB in zip(self.weights, deltaArr)]

        deltaArr.clear()

        #perceptron.clear()
        del perceptron[-1]  #   deletes bias to clear arr

        #   printing updated weights
        print("\nNEW [BIAS 'END']:")
        for x in self.weights:
            print(" ".join(map(str, x)))

        return self.weights

    def plotLearningCurve(self):
        x_data = []
        y_data = []
        x_data.extend([self.logged_error[i][0] for i in range(0, len(self.logged_error))])
        y_data.extend([self.logged_error[i][1] for i in range(0, len(self.logged_error))])
        fig, ax = plt.subplots()
        fig.suptitle("Leaning Rate")
        ax.set(xlabel = 'Epoch', ylabel = 'Squared Error')
        ax.plot(x_data, y_data, 'tab:green')
        plt.show()

    def training(self, inputs, unseen, bias, target, output):
        epoch = int(input("Amount of Epochs:"))
        step = int(input("Amount of Forward & Back Steps:"))

        i = 0

        #   Epochs
        for i in range(epoch):
            print(f"\nEpoch :: {i} || Target :: {target[0]}")

            #   Steps
            for j in range(step):
                print(f"\nStep: {j} || Dataset: {inputs[0]}") 
                self.forward(inputs[0], bias)
                self.back(inputs[0], target[0])
            
            for k in range(step):
                print(f"\nStep: {k} || Dataset: {inputs[1]}")
                self.forward(inputs[1], bias)
                self.back(inputs[1], target[0])

            for index in range(step):
                print(f"Step {index} || Dataset: {inputs[2]}")
                self.forward(inputs[2], bias)
                self.back(inputs[2], target[0])

            print(f"\nEpoch: {i} || Target: {target[1]}")
            
            for index in range(step):
                print(f"Step {index} || Dataset: {inputs[3]}")
                self.forward(inputs[3], bias)
                self.back(inputs[3], target[1])

            for index in range(step):
                print(f"Step {index} || Dataset: {inputs[4]}")
                self.forward(inputs[4], bias)
                self.back(inputs[4], target[1])

            for index in range(step):
                print(f"Step {index} || Dataset: {inputs[5]}")
                self.forward(inputs[5], bias)
                self.back(inputs[5], target[1])

            #   calculate mean
            x = [y ** 2 for y in self.errArr]
            self.total = (sum(x)) / len(self.errArr)

            self.logged_error.append([i, self.total])

        self.forward(unseen, bias)

        self.softmax(output)

        self.plotLearningCurve()

" MAIN "
def main():
    #   neuron inputs
    inputs = [[0.50, 1.00, 0.75], [1.00, 0.50, 0.75], [1.00, 1.00, 1.00], [-0.01, 0.50, 0.25], [0.50, -0.25, 0.13], [0.01, 0.02, 0.05]]

    unseen = [0.3, 0.7, 0.9]

    #   y1 & y2
    target = [[1, 0], [0, 1]]

    #   empty output
    output = []

    #   weights with bias
    weights = [[0.74, 0.8, 0.35, 0.9], [0.13, 0.4, 0.97, 0.45], [0.68, 0.10, 0.96, 0.36], [0.35, 0.50, 0.90, 0.98], [0.8, 0.13, 0.8, 0.92]]

    bias = [1]

    network = MLP_Network(weights, output)

    network.training(inputs, unseen, bias, target, output)

main()
