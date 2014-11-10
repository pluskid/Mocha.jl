using Mocha

use_cudnn = true
############################################################
# Prepare Random Data
############################################################
N = 10000
M = 20
P = 10

X = rand(M, N)
W = rand(M, P)
B = rand(P, 1)

Y = (W'*X .+ B)
Y = Y + 0.01*randn(size(Y))

############################################################
# Define network
############################################################
if use_cudnn
  sys = System(CuDNNBackend())
else
  sys = System(CPUBackend())
end
init(sys)

data_layer = MemoryDataLayer(batch_size=500, data=Array[X,Y])
weight_layer = InnerProductLayer(output_dim=P, tops=[:pred], bottoms=[:data])
loss_layer = SquareLossLayer(bottoms=[:pred, :label])

net = Net(sys, [loss_layer, weight_layer, data_layer])

############################################################
# Solve
############################################################
params = SolverParameters(regu_coef=0.0005, base_lr=0.01, momentum=0.9, max_iter=1000)
solver = SGD(params)
solve(solver, net)

learned_b = similar(B)
copy!(learned_b, net.states[2].b)

#println("$(learned_b)")
#println("$(B)")

shutdown(sys)
