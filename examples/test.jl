use_cudnn = false

using Mocha

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

net = Net("TEST", sys, [loss_layer, weight_layer, data_layer])

############################################################
# Solve
############################################################
lr_policy = LRPolicy.Staged(
  (6000, LRPolicy.Fixed(0.001)),
  (4000, LRPolicy.Fixed(0.0001)),
)
params = SolverParameters(regu_coef=0.0005, momentum=0.9, max_iter=10000, lr_policy=lr_policy)
solver = SGD(params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

solve(solver, net)

learned_b = similar(B)
copy!(learned_b, net.states[2].b)

#println("$(learned_b)")
#println("$(B)")

shutdown(sys)
