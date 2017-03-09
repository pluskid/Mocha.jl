

use_cuda = false

using Mocha


srand(12345678)

############################################################
# Prepare Random Data
############################################################
N = 5    # works with arbitrary minibatch size as long as
         # N == batch_size in MemoryDataLayer so it cycles through
         # and gets the same data during forward()
M = 10
P = 4

X = rand(M, N)
W = rand(M, P)
B = rand(P, 1)

Y = (W'*X .+ B)
Y = Y + 0.01*randn(size(Y))

############################################################
# Define network
############################################################
backend = CPUBackend()
init(backend)

data_layer = MemoryDataLayer(batch_size=N, data=Array[X,Y])

w1 = InnerProductLayer(neuron=Neurons.Sigmoid(), name="ip1",output_dim=20, tops=[:a], bottoms=[:data])
w2 = InnerProductLayer(neuron=Neurons.Identity(), name="ip2",output_dim=4, tops=[:b], bottoms=[:a])
loss_layer = SquareLossLayer(name="loss", bottoms=[:b, :label] )


net = Net("TEST", backend, [w1,w2, loss_layer, data_layer])

# epsilon:     milage may vary 1e-4 - 1e-8
# digit:       compare this many digits to check for 'identity'
# visualize:   prints out correct and failed positions of gradients
# return value: false if any gradients fails, true if all passes
test_gradients(net, epsilon=1e-8, digit=6, visual=true )

shutdown(backend)


