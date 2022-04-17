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

#   weights with bias
weights = [[0.74, 0.8, 0.35, 0.9], [0.13, 0.4, 0.97, 0.45], [0.68, 0.10, 0.96, 0.36], [0.35, 0.50, 0.90, 0.98], [0.8, 0.13, 0.8, 0.92]]
#   w47 = 0.35, w48 = 0.8, w57 = 0.50, w58 = 0.13, w67 = 0.90, w68 = 0.8

bias = [1]

" MAIN "

_ann = ann.Network(weights, target)
#_ann.forward()
#_ann.back()

for i in range(2):
    print(f"Step: {i}")
    _ann.forward(inputs, bias)
    _ann.back(inputs)




