import math
import ann

'''
//  Backpropagation algorithm
//  -> Apply it to the network architecture and data below..
//      -> Inputs X and outputs Y
//  -> No in-built libraries
'''
#   x - inputs
inputs = [0.50, 1.00, 0.75]
#   output for this example ^ -> first row, y1
target = [1, 0]

#   w - weights with added bias w4-w6 [last element in an array]
weights = [[0.74, 0.13, 0.68], [0.8, 0.4, 0.10], [0.35, 0.97, 0.96]]
#   w47 = 0.35, w48 = 0.8, w57 = 0.50, w58 = 0.13, w67 = 0.90, w68 = 0.8

#   needs adding -> not working
bias = [[0.9], [0.45], [0.36]]

" MAIN "

_ann = ann.Network()




