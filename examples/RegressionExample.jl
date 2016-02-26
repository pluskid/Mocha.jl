############################################################
# set up environment
############################################################
ENV["MOCHA_USE_NATIVE_EXT"] = "true"
ENV["OMP_NUM_THREADS"] = 1
blas_set_num_threads(4)
using Mocha
srand(12345678)
backend = DefaultBackend()
init(backend)
snapshot_dir = "regression_example_snapshots"
############################################################
# Load the data (already pre-processed)
############################################################
n = 100000 # size of training sample
nn = 10000
X = randn(10,n+nn)
Y = [sin(X[1,:]+X[2,:]); cos(X[3,:]+X[4,:])]    # two outputs, only first 4 inputs are relevant
Y = Y + 0.5*randn(size(Y))                      # add noise: a good fit would give RMSE=0.5
# split the training and testing samples
YT = Y[:,1:nn]      # test
XT = X[:,1:nn]      # test
Y = Y[:,nn+1:end]   # training 
X= X[:,nn+1:end]    # training
############################################################
# Define network
############################################################
# specify sizes of layers
Layer1Size = 100
Layer2Size = 10
# create the network
data = MemoryDataLayer(batch_size=500, data=Array[X,Y])
h1 = InnerProductLayer(name="ip1",neuron=Neurons.Tanh(), output_dim=Layer1Size, tops=[:pred1], bottoms=[:data])
h2 = InnerProductLayer(name="ip2",neuron=Neurons.Tanh(), output_dim=Layer2Size, tops=[:pred2], bottoms=[:pred1])
output = InnerProductLayer(name="aggregator", output_dim=size(Y,1), tops=[:output], bottoms=[:pred2] )
loss_layer = SquareLossLayer(name="loss", bottoms=[:output, :label])
common_layers = [h1,h2,output]
net = Net("regression-train", backend, [data, common_layers, loss_layer])
# create the validation network
datatest = MemoryDataLayer(batch_size=10000, data=Array[XT,YT])
accuracy = RegressionAccuracyLayer(name="acc", bottoms=[:output, :label])
test_net = Net("regression-test", backend, [datatest, common_layers, accuracy])
validate_performance = ValidationPerformance(test_net)
############################################################
# Solve
############################################################
lr_policy = LRPolicy.DecayOnValidation(0.02, "test-accuracy-accuracy", 0.5)
method = SGD()
params = make_solver_parameters(method, regu_coef=0.0001, mom_policy=MomPolicy.Fixed(0.9), max_iter=100000, lr_policy=lr_policy, load_from=snapshot_dir)
solver = Solver(method, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)
add_coffee_break(solver, Snapshot(snapshot_dir), every_n_iter=5000)
add_coffee_break(solver, validate_performance, every_n_iter=1000)
# link the decay-on-validation policy with the actual performance validator
setup(lr_policy, validate_performance, solver)
solve(solver, net)
Mocha.dump_statistics(solver.coffee_lounge, get_layer_state(net, "loss"), true)
destroy(net)
destroy(test_net)
shutdown(backend)
