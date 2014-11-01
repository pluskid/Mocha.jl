using Mocha

############################################################
# Prepare Random Data
############################################################
N = 10000
M = 20
P = 1

X = rand(M, N)
W = rand(M, P)
B = rand(P, 1)

Y = (W'*X .+ B)
Y = Y + 0.01*randn(size(Y))

############################################################
# Define network
############################################################
sys = System(CuDNNBackend(), 0.0005, 0.01, 0.9, 5000)
init(sys)

data_layer = MemoryDataLayer(; batch_size=100, data=Array[X,Y])
weight_layer = InnerProductLayer(; output_dim=M, tops=String["pred"], bottoms=String["data"])
loss_layer = SquareLossLayer(; bottoms=String["pred", "label"])

net = Net(sys, [loss_layer, weight_layer, data_layer])

############################################################
# Solve
############################################################
solver = SGD()
solve(solver, net)
shutdown(sys)
