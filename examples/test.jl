use_cuda = false

using Mocha
srand(12345678)

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
if use_cuda
  backend = GPUBackend()
else
  backend = CPUBackend()
end
init(backend)

data_layer = MemoryDataLayer(batch_size=500, data=Array[X,Y])
weight_layer = InnerProductLayer(name="ip",output_dim=P, tops=[:pred], bottoms=[:data])
loss_layer = SquareLossLayer(name="loss", bottoms=[:pred, :label])

net = Net("TEST", backend, [loss_layer, weight_layer, data_layer])
println(net)

############################################################
# Solve
############################################################
lr_policy = LRPolicy.Staged(
  (6000, LRPolicy.Fixed(0.001)),
  (4000, LRPolicy.Fixed(0.0001)),
)
method = SGD()
params = make_solver_parameters(method, regu_coef=0.0005, mom_policy=MomPolicy.Fixed(0.9), max_iter=10000, lr_policy=lr_policy)
solver = Solver(method, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

solve(solver, net)

learned_b = similar(B)
copy!(learned_b, net.states[2].b)

#println("$(learned_b)")
#println("$(B)")

Mocha.dump_statistics(solver.coffee_lounge, get_layer_state(net, "loss"), true)

shutdown(backend)
