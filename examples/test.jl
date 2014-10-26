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

Y = X*W .+ B + 0.1*randn(N, P)

############################################################
# Define network
############################################################
data_layer = MemoryDataLayer(; batch_size=100, data=Array[X,Y])
weight_layer = InnerProductLayer(; output_dim=P, tops=String["pred"], bottoms=String["data"])
loss_layer = SquareLossLayer(; bottoms=String["pred", "label"])

net = Net([loss_layer, weight_layer, data_layer])

############################################################
# Solve
############################################################
solver = SGD(100, 0.01)
solve(solver, net)
