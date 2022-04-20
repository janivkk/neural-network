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

#   work on it
""" print(inputs[0])
network.forward(inputs[0], bias)
network.back(inputs[0], target[0], bias)
print(inputs[1])
network.forward(inputs[1], bias)
network.back(inputs[1], target[0], bias)
print(inputs[2])
network.forward(inputs[2], bias) """
#network.squareEr(target[0], output)

def main():
    print("Backpropagation Algorithm (ID: 25085325) => Neural Networks")

    epoch = 1
    step = 1

    #   Epoch
    for i in range(epoch):
        print(f"\nEpoch: {i} || Target: {target[0]}")

        #   Steps
        for j in range(step):
            print(f"\nStep: {j} || Dataset: {inputs[0]}") 
            network.forward(inputs[0], bias)
            network.back(inputs[0], target[0])
        
        for k in range(step):
            print(f"\nStep: {k} || Dataset: {inputs[1]}")
            network.forward(inputs[1], bias)
            network.back(inputs[1], target[0])

        for index in range(step):
            print(f"Step {index} || Dataset: {inputs[2]}")
            network.forward(inputs[2], bias)
            network.back(inputs[2], target[0])

        print(f"\nEpoch: {i} || Target: {target[1]}")
        
        for index in range(step):
            print(f"Step {index} || Dataset: {inputs[3]}")
            network.forward(inputs[3], bias)
            network.back(inputs[3], target[1])

        for index in range(step):
            print(f"Step {index} || Dataset: {inputs[4]}")
            network.forward(inputs[4], bias)
            network.back(inputs[4], target[1])

        for index in range(step):
            print(f"Step {index} || Dataset: {inputs[5]}")
            network.forward(inputs[5], bias)
            network.back(inputs[5], target[1])    

    print("Probability Distribution [Y] of 0.3, 0.7 & 0.9")
    network.forward(unseen, bias)

    network.softmax(output)

main()







