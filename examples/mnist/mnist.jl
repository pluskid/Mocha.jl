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

using Mocha

data_layer = HDF5DataLayer(source=source_fns[1], batch_size=64)
conv_layer = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
pool_layer = PoolingLayer(kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer  = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer  = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer = SoftmaxLossLayer(bottoms=[:ip2,:label])

#sys = System(CuDNNBackend())
sys = System(CPUBackend())
init(sys)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
net = Net(sys, [data_layer, common_layers..., loss_layer])

params = SolverParameters(max_iter=20, regu_coef=0.0005, base_lr=0.01, momentum=0.9,
    lr_policy=LRPolicy.Inv(0.0001, 0.75))
solver = SGD(params)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# show performance on test data every 1000 iterations
data_layer_test = HDF5DataLayer(source=source_fns[2], batch_size=100)
acc_layer = AccuracyLayer(bottoms=[:ip2, :label])
test_net = Net(sys, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

# save snapshots every 5000 iterations
add_coffee_break(solver,
    Snapshot("snapshots", auto_load=true, also_load_solver_state=false),
    every_n_iter=5000)

blas_set_num_threads(16)
#solve(solver, net)
Profile.init(int(1e8), 0.001)
@profile solve(solver, net)
open("profile.txt", "w") do out
  Profile.print(out)
end

shutdown(sys)
