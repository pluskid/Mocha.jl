ENV["MOCHA_USE_CUDA"] = "true"

using Mocha

data_tr_layer = HDF5DataLayer(source="data/train.txt", batch_size=100)
data_tt_layer = HDF5DataLayer(source="data/test.txt", batch_size=100)

conv1_layer = ConvolutionLayer(name="conv1", n_filter=32, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=GaussianInitializer(std=0.0001),
    bottoms=[:data], tops=[:conv1])
pool1_layer = PoolingLayer(kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
    bottoms=[:conv1], tops=[:pool1])
norm1_layer = LRNLayer(kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel()
    bottoms=[:pool1], tops=[:norm1])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=32, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=GaussianInitializer(std=0.01),
    bottoms=[:norm1], tops=[:conv2], neuron=Neurons.ReLU())
pool2_layer = PoolingLayer(kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
    bottoms=[:conv2], tops=[:pool2])
norm2_layer = LRNLayer(kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
    bottoms=[:pool2], tops=[:norm2])
conv3_layer = ConvolutionLayer(name="conv3", n_filter=64, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=GaussianInitializer(std=0.01),
    bottoms=[:norm2], tops=[:conv3], neuron=Neurons.ReLU())
pool3_layer = PoolingLayer(kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
    bottoms=[:conv3], tops=[:pool3])
ip1_layer   = InnerProductLayer(output_dim=10, weight_init=GaussianInitializer(std=0.01),
    weight_regu=L2Regu(250), bottoms=[:pool3], tops=[:ip1])

loss_layer  = SoftmaxLossLayer(bottoms=[:ip1, :label])
acc_layer   = AccuracyLayer(bottoms=[:ip1, label])

common_layers = [conv1_layer, pool1_layer, norm1_layer, conv2_layer, pool2_layer, norm2_layer,
                 conv3_layer, pool3_layer, ip1_layer]

sys = System(CPUBackend())
init(sys)

net = Net(sys, [data_tr_layer, common_layers..., loss_layer])
