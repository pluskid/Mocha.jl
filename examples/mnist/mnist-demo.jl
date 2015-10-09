#############################################################
# This file is primarily for the purpose of demostrating
# various features:
#
# * DecayOnValidation learning rate policy scheduling
# * User defined Coffee
#
#############################################################
using Mocha
srand(12345678)

data_layer  = AsyncHDF5DataLayer(name="train-data", source="data/train.txt", batch_size=64, shuffle=true)
conv_layer  = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
pool_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer   = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

backend = DefaultBackend()
init(backend)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
net = Net("MNIST-train", backend, [data_layer, common_layers..., loss_layer])

exp_dir = "snapshots-$(Mocha.default_backend_type)"

data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])
validate_performance = ValidationPerformance(test_net)
lr_policy = LRPolicy.DecayOnValidation(1.e-2, "test-accuracy-accuracy", 0.5)

method = SGD()
params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=lr_policy,
                                load_from=exp_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# a dummy demo of user-defined coffe break
type DummyCoffee <: Coffee
end
function Mocha.enjoy(lounge::CoffeeLounge, coffee::DummyCoffee, ::Net, ::SolverState)
  println("See also src/coffee/*.jl for implementation of built-in coffee breaks")
end
add_coffee_break(solver, DummyCoffee(), every_n_iter=500)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
add_coffee_break(solver, validate_performance, every_n_iter=1000)

# link the decay-on-validation policy with the actual performance validator
setup(lr_policy, validate_performance, solver)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
