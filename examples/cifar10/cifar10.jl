ENV["MOCHA_USE_CUDA"] = "true"

using Mocha

data_layer = HDF5DataLayer(source="data/train.txt", batch_size=100)
conv1_layer = ConvolutionLayer(name="conv1", n_filter=32, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=GaussianInitializer(std=0.0001),
    bottoms=[:data], tops=[:conv1])
pool1_layer = PoolingLayer(kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
    bottoms=[:conv1], tops=[:pool1])
