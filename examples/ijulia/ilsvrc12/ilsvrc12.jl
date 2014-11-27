ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha
img_width, img_height, img_channels = (256, 256, 3)
crop_size = (227, 227)
batch_size = 1

layers = [
  MemoryDataLayer(name="data", tops=[:data], batch_size=batch_size,
      transformers=[(:data, DataTransformers.SubMean(mean_file="model/ilsvrc_2012_mean.hdf5")),
                    #(:data, DataTransformers.Scale(scale=1.0/255))
                    ],
      data = Array[zeros(img_width, img_height, img_channels, batch_size)])
  CropLayer(name="crop", tops=[:cropped], bottoms=[:data], crop_size=crop_size)
  ConvolutionLayer(name="conv1", tops=[:conv1], bottoms=[:cropped],
      kernel=(11,11), stride=(4,4), n_filter=96, neuron=Neurons.ReLU())
  pool1_layer = PoolingLayer(name="pool1", tops=[:pool1], bottoms=[:conv1],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  LRNLayer(name="norm1", tops=[:norm1], bottoms=[:pool1],
      kernel=5, scale=0.0001, power=0.75)
  ConvolutionLayer(name="conv2", tops=[:conv2], bottoms=[:norm1],
      kernel=(5,5), pad=(2,2), n_filter=256, n_group=2, neuron=Neurons.ReLU())
  PoolingLayer(name="pool2", tops=[:pool2], bottoms=[:conv2],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  LRNLayer(name="norm2", tops=[:norm2], bottoms=[:pool2],
      kernel=5, scale=0.0001, power=0.75)
  ConvolutionLayer(name="conv3", tops=[:conv3], bottoms=[:norm2],
      kernel=(3,3), pad=(1,1), n_filter=384, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv4", tops=[:conv4], bottoms=[:conv3],
      kernel=(3,3), pad=(1,1), n_filter=384, n_group=2, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv5", tops=[:conv5], bottoms=[:conv4],
      kernel=(3,3), pad=(1,1), n_filter=256, n_group=2, neuron=Neurons.ReLU())
  PoolingLayer(name="pool5", tops=[:pool5], bottoms=[:conv5],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  InnerProductLayer(name="fc6", tops=[:fc6], bottoms=[:pool5],
      output_dim=4096, neuron=Neurons.ReLU())
  InnerProductLayer(name="fc7", tops=[:fc7], bottoms=[:fc6],
      output_dim=4096, neuron=Neurons.ReLU())
  InnerProductLayer(name="fc8", tops=[:fc8], bottoms=[:fc7],
      output_dim=1000)
  SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:fc8])
]

sys = System(CPUBackend())
init(sys)

net = Net("imagenet", sys, layers)

using HDF5
h5open("model/bvlc_reference_caffenet.hdf5", "r") do h5
  load_network(h5, net)
end
