#ENV["MOCHA_USE_NATIVE_EXT"] = "true"
#ENV["OMP_NUM_THREADS"] = 1
#blas_set_num_threads(1)
ENV["MOCHA_USE_CUDA"] = "true"

using Mocha

############################################################
# This is an example script for training a fully connected
# network with dropout on mnist.
#
# The network size is 784-1200-1200-10 with ReLU units
# in the hidden layers and a softmax output layer.
# The parameters for training the network were chosen
# to reproduce the results from the original dropout paper:
# http://arxiv.org/abs/1207.0580
# and the corresponding newer JMLR paper:
# http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
#
# Our parameters slightly differ. This is mainly due to the
# fact that in the original dropout paper the weights are scaled
# by 0.5 after training whereas we scale them by 2 during training.
#
# The settings in this script should currently produce a model that
# gets 94 errors (or 99.06 % accuracy) on the test set
# if you run it for the whole 2000 epochs (=600*2000 steps).
# This is slightly better than the results of the JMLR paper.
# This difference is likely due to slight differences in the
# learning parameters. Also note that our hyperparameters
# are not chosen using a validation set, as one would do
# for a paper. If your hardware and cuda versions differ
# from the setup we used for intial testing your results might
# also slightly vary due to floating point inaccuracies.
############################################################


# fix the random seed to make results reproducable
srand(12345678)

data_layer  = HDF5DataLayer(name="train-data", source="data/train.txt", batch_size=100)
# each fully connected layer uses a ReLU activation and a constraint on the L2 norm of the weights
fc1_layer   = InnerProductLayer(name="fc1", output_dim=1200, neuron=Neurons.ReLU(),
                                weight_init = GaussianInitializer(std=0.01),
                                #weight_cons = L2Cons(4.5),
                                bottoms=[:data], tops=[:fc1])
fc2_layer   = InnerProductLayer(name="fc2", output_dim=1200, neuron=Neurons.ReLU(),
                                weight_init = GaussianInitializer(std=0.01),
                                weight_cons = L2Cons(4.5),
                                bottoms=[:fc1], tops=[:fc2])
fc3_layer   = InnerProductLayer(name="out", output_dim=10, bottoms=[:fc2],
                                weight_init = ConstantInitializer(0),
                                weight_cons = L2Cons(4.5),
                                tops=[:out])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:out,:label])

# setup dropout for the different layers
# we use 20% dropout on the inputs and 50% dropout in the hidden layers
# as these values were previously found to be good defaults
drop_input  = DropoutLayer(name="drop_in", bottoms=[:data], ratio=0.2)
drop_fc1 = DropoutLayer(name="drop_fc1", bottoms=[:fc1], ratio=0.5)
drop_fc2  = DropoutLayer(name="drop_fc2", bottoms=[:fc2], ratio=0.5)

backend = GPUBackend()
init(backend)

common_layers = [fc1_layer, fc2_layer, fc3_layer]
drop_layers = [drop_input, drop_fc1, drop_fc2]
# put training net together, note that the correct ordering will automatically be established by the constructor
net = Net("MNIST-train", backend, [data_layer, common_layers..., drop_layers..., loss_layer])

base_dir = "snapshots_dropout_fc"
# we let the learning rate decrease by 0.998 in each epoch (=600 batches of size 100)
# and let the momentum increase linearly from 0.5 to 0.9 over 500 epochs
# which is equivalent to an increase step of 0.0008
# training is done for 2000 epochs
params = SolverParameters(max_iter=600*2000, regu_coef=0.0,
                          mom_policy=MomPolicy.Linear(0.5, 0.0008, 600, 0.9),
                          lr_policy=LRPolicy.Step(0.1, 0.998, 600),
                          load_from=base_dir)
solver = SGD(params)

setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

# show performance on test data every 600 iterations (one epoch)
data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:out, :label], report_error=true)
test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=600)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
