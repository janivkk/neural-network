'''
SUBJECT TO CHANGE

THIS FILE SAVED FOR IMPLEMENTATION OF ANN CLASS
'''

#   workshop task -> implement slide 30 with multiple weights

import math

#   inputs
x = [0, 1, 1]
bias = 1
#   weights [2d]
weights = [[0.5, -0.2, 0.5], [0.1, 0.2, 0.3], [0.7, 0.6, 0.2], [0.9, 0.8, 0.4]]

#   output
output = []
#   desired output
desired_output = [1, 0]

#   first Hidden's layer Sum Array
netHiddenFirst = []

#   net array - final
netarray = []

#   activational function / sigmoid
sigmoid = []

#   softmax
softmaxArr = []

#   error output
error_output = []

#   hidden output
hidden_output = []

#   test
testarr = []

def forwardStep(): #    slide 30 net4,5 o4,5
    #   first hidden layer
    for neuron_weights in weights[:2]: #    chooses only 2 rows
        net = 0
        for x_input, weight in zip(x, neuron_weights):
            net += x_input * weight
        netHiddenFirst.append(net)
    print(f"net 4,5: {netHiddenFirst}") # works

    #   sigmoid w/o activating the function or firing up a neuron
    for nets in range(len(netHiddenFirst)):
        sigmoid.append(1 / (1 + (math.exp(-netHiddenFirst[nets]))))
    #   adds bias to the sigmoid as per slide example
    sigmoid.insert(len(sigmoid), bias)
    print(f"o 4,5 + bias: {sigmoid}") #    works

    for neuron_weights in weights[2:]:
        net = 0
        for s_weight, w in zip(sigmoid, neuron_weights):
            net += w * s_weight
        netarray.append(net)
    print(f"net 6,7 & o 6,7: {netarray}") # works

def backwardStep():
    #   output errors
    for i in range(len(desired_output)):
        #   formula
        temp = 0
        temp = desired_output[i] - netarray[i]
        error_output.append(temp)
    print(f"Output errors: {error_output}")

    a = [i[0] for i in weights[2:]]

    for i in range(len(error_output)):
        temp = 0
        for j in a:
            #pass
            temp += a[i > j] * error_output[i] 
        print(temp)    
        
    #   hidden errors
    for j in [i[0] for i in weights[2:]]:
        temp = 0
        tempArr = []
        for k in range(len(error_output)):
            temp += j * error_output[k]
            tempArr.append(temp)
    #print(f"temp: {tempArr}")
        #   formula - e.g.,
        #   o[i] * (1 - o[i]) * [(0.7 * 0.0244) + (0.9 * -1.414)]...
        #print(f"Hidden layer: {hidden_output}")

    #print(f"weights: {weights[2::1]}")

    #-1.25552

    #   weight update

#   training

#   softmax function...?

#   output errors then hidden errors in layer 1

print("Forward Step:")
forwardStep()
print("Backward Step:")
backwardStep()    

