import math
import ann

'''
//  Backpropagation algorithm
//  -> Apply it to the network architecture and data below..
//      -> Inputs X and outputs Y
//  -> No in-built libraries
'''
#   bias is the last [appended] element in weights or inputs

#   x - inputs
inputs = [0.50, 1.00, 0.75]
#   output for this example ^ -> first row, y1
target = [1, 0]

#   w - weights with added bias w4-w6 [last element in an array] no bias
#weights = [[0.74, 0.8, 0.35], [0.13, 0.4, 0.97], [0.68, 0.10, 0.96], [0.35, 0.50, 0.90], [0.8, 0.13, 0.8]]

#   weights with bias
weights = [[0.74, 0.8, 0.35, 0.9], [0.13, 0.4, 0.97, 0.45], [0.68, 0.10, 0.96, 0.36], [0.35, 0.50, 0.90], [0.8, 0.13, 0.8]]
#   w47 = 0.35, w48 = 0.8, w57 = 0.50, w58 = 0.13, w67 = 0.90, w68 = 0.8

#   bias
bias = [[0.9], [0.45], [0.36], [0.98], [0.92]] #    up to 0.36 so :3

biasTest = [1]

netArr = []

" MAIN "

_ann = ann.Network(inputs, weights, biasTest)

_ann.forward()

def test(): 
    inputs.extend(biasTest)

    for neuron_weights in weights[:3]:
        net = 0
        for x_input, weight in zip(inputs, neuron_weights):
            net += x_input * weight
        netArr.append(net)
    print(f"net 1, 2, 3: {netArr}")

#test()




