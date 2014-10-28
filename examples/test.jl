using Mocha

############################################################
# Prepare Random Data
############################################################
N = 1000
M = 20
P = 10

X = rand(N, M)
W = rand(M, P)
B = rand(1, P)

Y = (X*W .+ B).^2

############################################################
# Define network
############################################################
sys = System(CPU(), 0.0005, 0.01, 0.9, 5000)

data_layer = MemoryDataLayer(; batch_size=100, data=Array[X,Y])
weight_layer = InnerProductLayer(; output_dim=P, tops=String["pred"], bottoms=String["data"], neuron = Neurons.ReLU())
weight_layer2 = InnerProductLayer(; output_dim=P, tops=String["pred2"], bottoms=String["pred"], neuron = Neurons.ReLU())
loss_layer = SquareLossLayer(; bottoms=String["pred2", "label"])

net = Net(sys, [loss_layer, weight_layer, weight_layer2, data_layer])

############################################################
# Solve
############################################################
solver = SGD()
solve(solver, net)
