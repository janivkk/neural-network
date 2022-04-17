'''
//  Backpropagation algorithm
//  -> Apply it to the network architecture and data below..
//      -> Inputs X and outputs Y
//  -> No in-built libraries
'''
" LIBRARIES "
import ann

" MAIN "
#   x - inputs
inputs = [[0.50, 1.00, 0.75], [1.00, 0.50, 0.75], [1.00, 1.00, 1.00]]

#   output for this example ^ -> first row, y1
target = [1, 0]

#   empty output
output = []


#   weights with bias
weights = [[0.74, 0.8, 0.35, 0.9], [0.13, 0.4, 0.97, 0.45], [0.68, 0.10, 0.96, 0.36], [0.35, 0.50, 0.90, 0.98], [0.8, 0.13, 0.8, 0.92]]
#   w47 = 0.35, w48 = 0.8, w57 = 0.50, w58 = 0.13, w67 = 0.90, w68 = 0.8

bias = [1]

network = ann.Network(weights, target, output)
#_ann.forward()
#_ann.back()

def run():
    for i in range(1):
        print(f"Epoch: {i}\n")
        for j in range(5):
            print(f"Step: {j}")
            network.forward(inputs[0], bias)
            network.back(inputs[0])

        for k in range(5):
            print(f"Step: {k}")
            network.forward(inputs[1], bias)
            network.back(inputs[1])

    network.softmax(output)

run()







