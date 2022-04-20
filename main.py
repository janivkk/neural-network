'''
//  Backpropagation algorithm
//  -> Apply it to the network architecture and data below..
//      -> Inputs X and outputs Y
//  -> No in-built libraries
'''
" LIBRARIES "
import mlp_network

" MAIN "
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

network = mlp_network.MLP_Network(weights, output)

network.training(inputs, unseen, bias, target, output)
