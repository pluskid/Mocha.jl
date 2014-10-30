using Mocha

############################################################
# Prepare Random Data
############################################################
N = 10000
M = 20
P = 1

X = rand(N, M)
W = rand(M, P)
B = rand(1, P)

Y = ((X*W .+ B)).^2
m = mean(Y)
Y[Y .> m] = 2
Y[Y .<= m] = 1

############################################################
# Define network
############################################################
sys = System(CPUBackend(), 0.0005, 0.01, 0.9, 5000)

data_layer = MemoryDataLayer(; batch_size=100, data=Array[X,Y])
weight_layer = InnerProductLayer(; output_dim=M, tops=String["pred"], bottoms=String["data"], neuron = Neurons.Sigmoid())
weight_layer2 = InnerProductLayer(; output_dim=2, tops=String["pred2"], bottoms=String["pred"], neuron = Neurons.Identity())
loss_layer = SoftmaxLossLayer(; bottoms=String["pred2", "label"])

net = Net(sys, [loss_layer, weight_layer, weight_layer2, data_layer])

############################################################
# Solve
############################################################
solver = SGD()
solve(solver, net)
