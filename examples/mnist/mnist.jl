train_fn = "data/train.hdf5"
train_source_fn = "data/train.txt"
if !isfile(train_fn)
  println("Data not found, use get-mnist.sh to generate HDF5 data")
  exit(1)
else
  open(train_source_fn, "w") do s
    println(s, train_fn)
  end
end

using Mocha

data_layer = HDF5DataLayer(source=train_source_fn, batch_size=64)
conv_layer = ConvolutionLayer(n_filter=20, kernel=(5,5), bottoms=String["data"], tops=String["conv"])
pool_layer = PoolingLayer(kernel=(2,2), stride=(2,2), bottoms=String["conv"], tops=String["pool"])
fc1_layer  = InnerProductLayer(output_dim=500, neuron=Neurons.ReLU(), bottoms=String["pool"], tops=String["ip1"])
fc2_layer  = InnerProductLayer(output_dim=10, bottoms=String["ip1"], tops=String["ip2"])
#fc2_layer  = InnerProductLayer(output_dim=10, bottoms=String["data"], tops=String["ip2"])
loss_layer = SoftmaxLossLayer(bottoms=String["ip2","label"])

#sys = System(CuDNNBackend(), 0.0005, 0.01, 0.9, 10000)
sys = System(CuDNNBackend(), 0.0005, 0.01, 0.9, 1000)
init(sys)

net = Net(sys, [data_layer, conv_layer, pool_layer, fc1_layer, fc2_layer, loss_layer])
#net = Net(sys, [data_layer, fc2_layer, loss_layer])

solver = SGD()
solve(solver, net)

shutdown(sys)
