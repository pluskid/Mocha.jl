hdf5_fns = ["data/train.hdf5", "data/test.hdf5"]
source_fns = ["data/train.txt", "data/test.txt"]
for i = 1:length(hdf5_fns)
  if !isfile(hdf5_fns[i])
    println("Data not found, use get-mnist.sh to generate HDF5 data")
    exit(1)
  else
    open(source_fns[i], "w") do s
      println(s, hdf5_fns[i])
    end
  end
end

#ENV["MOCHA_USE_NATIVE_EXT"] = "true"
#ENV["OMP_NUM_THREADS"] = 1
#blas_set_num_threads(1)
ENV["MOCHA_USE_CUDA"] = "true"

using Mocha

# fix the random seed to make results reproducable
srand(12345678)

data_layer  = HDF5DataLayer(name="train-data", source=source_fns[1], batch_size=100)
fc1_layer   = InnerProductLayer(name="fc1", output_dim=1200, neuron=Neurons.ReLU(), weight_init = GaussianInitializer(std=0.01), bottoms=[:data], tops=[:fc1])
fc2_layer   = InnerProductLayer(name="fc2", output_dim=1200, neuron=Neurons.ReLU(), weight_init = GaussianInitializer(std=0.01), bottoms=[:fc1], tops=[:fc2])
fc3_layer   = InnerProductLayer(name="out", output_dim=10, bottoms=[:fc2], weight_init = ConstantInitializer(0), tops=[:out])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:out,:label])

# setup dropout for the different layers
drop_input  = DropoutLayer(name="drop_in", bottoms=[:data], ratio=0.2)
drop_fc1 = DropoutLayer(name="drop_fc1", bottoms=[:fc1], ratio=0.5)
drop_fc2  = DropoutLayer(name="drop_fc2", bottoms=[:fc2], ratio=0.5)

sys = System(CuDNNBackend())
#sys = System(CPUBackend())
init(sys)

common_layers = [fc1_layer, fc2_layer, fc3_layer]
drop_layers = [drop_input, drop_fc1, drop_fc2]
# put training net together, note that the correct ordering will automatically be established by the constructor
net = Net("MNIST-train", sys, [data_layer, common_layers..., drop_layers..., loss_layer])

params = SolverParameters(max_iter=600*1000, regu_coef=0.0, mom_policy=MomPolicy.Linear(0.5, 0.0008, 600, 0.9), 
                          #mom_policy=MomPolicy.Step(0.5, 1.0012, 600, 0.9),
                          lr_policy=LRPolicy.Step(0.1, 0.998, 600))
solver = SGD(params)

base_dir = "snapshots_dropout_fc"
# save snapshots every 5000 iterations
add_coffee_break(solver,
                 Snapshot(base_dir, auto_load=true),
                 every_n_iter=5000)
                 
# show performance on test data every 600 iterations (one epoch)
# also log evrything using the AccumulateStatistics module
data_layer_test = HDF5DataLayer(name="test-data", source=source_fns[2], batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:out, :label], report_error=true)
test_net = Net("MNIST-test", sys, [data_layer_test, common_layers..., acc_layer])
stats = AccumulateStatistics([ValidationPerformance(test_net), TrainingSummary()], try_load = true, save = true, fname = "$(base_dir)/statistics.h5")
add_coffee_break(solver, stats, every_n_iter=600)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(sys)
