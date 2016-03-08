using Mocha

data_tr_layer = AsyncHDF5DataLayer(name="data-train", source="data/train.txt", batch_size=32, shuffle=@windows ? false : true)
data_tt_layer = AsyncHDF5DataLayer(name="data-test", source="data/test.txt", batch_size=32)
conv1_layer = ConvolutionLayer(name="conv1", n_filter=96, kernel=(11,11),
    stride=(4,4), filter_init=GaussianInitializer(std=0.01), neuron=Neurons.ReLU(),
    bottoms=[:data], tops=[:conv1])
norm1_layer = LRNLayer(name="norm1", kernel=5, scale=1e-4, power=0.75, mode=LRNMode.WithinChannel(),
    bottoms=[:conv1], tops=[:norm1])
pool1_layer = PoolingLayer(name="pool1", kernel=(3,3), stride=(2,2), pooling=Pooling.Max(),
    bottoms=[:norm1], tops=[:pool1])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=256, kernel=(5,5), pad=(2,2), n_group=2,
    filter_init=GaussianInitializer(std=0.01), bias_init=ConstantInitializer(0.1),
    bottoms=[:pool1], tops=[:conv2], neuron=Neurons.ReLU())
norm2_layer = LRNLayer(name="norm2", kernel=5, scale=1e-4, power=0.75, mode=LRNMode.WithinChannel(),
    bottoms=[:conv2], tops=[:norm2])
pool2_layer = PoolingLayer(name="pool2", kernel=(3,3), stride=(2,2), pooling=Pooling.Max(),
    bottoms=[:norm2], tops=[:pool2])
conv3_layer = ConvolutionLayer(name="conv3", n_filter=384, kernel=(3,3), pad=(1,1),
    stride=(1,1), filter_init=GaussianInitializer(std=0.01),
    bottoms=[:pool2], tops=[:conv3], neuron=Neurons.ReLU())
conv4_layer = ConvolutionLayer(name="conv4", n_filter=384, kernel=(3,3), pad=(1,1), n_group=2,
    stride=(1,1), filter_init=GaussianInitializer(std=0.01), bias_init=ConstantInitializer(0.1),
    bottoms=[:conv3], tops=[:conv4], neuron=Neurons.ReLU())
conv5_layer = ConvolutionLayer(name="conv5", n_filter=256, kernel=(3,3), pad=(1,1), n_group=2,
    stride=(1,1), filter_init=GaussianInitializer(std=0.01), bias_init=ConstantInitializer(0.1),
    bottoms=[:conv4], tops=[:conv5], neuron=Neurons.ReLU())
pool5_layer = PoolingLayer(name="pool5", kernel=(3,3), pad=(1,1), stride=(2,2), pooling=Pooling.Max(),
    bottoms=[:conv5], tops=[:pool5])
fc6_layer   = InnerProductLayer(name="fc6", output_dim=4096, weight_init=GaussianInitializer(std=0.005),
    bias_init=ConstantInitializer(0.1), bottoms=[:pool5], tops=[:fc6], neuron=Neurons.ReLU())
drop6_layer  = DropoutLayer(name="drop6", bottoms=[:fc6], ratio=0.5)
fc7_layer   = InnerProductLayer(name="fc7", output_dim=4096, weight_init=GaussianInitializer(std=0.005),
    bias_init=ConstantInitializer(0.1), bottoms=[:fc6], tops=[:fc7], neuron=Neurons.ReLU())
drop7_layer  = DropoutLayer(name="drop7", bottoms=[:fc7], ratio=0.5)
fc8_layer   = InnerProductLayer(name="fc8", output_dim=1000, weight_init=GaussianInitializer(std=0.01),
    bias_init=ConstantInitializer(0), bottoms=[:fc7], tops=[:fc8])

loss_layer  = SoftmaxLossLayer(name="softmax", bottoms=[:fc8, :label])
acc_layer   = AccuracyLayer(name="accuracy", bottoms=[:fc8, :label])

common_layers = [conv1_layer, norm1_layer, pool1_layer, 
                 conv2_layer, norm2_layer, pool2_layer,
                 conv3_layer,
                 conv4_layer,
                 conv5_layer, pool5_layer,
                 fc6_layer, #drop6_layer,
                 fc7_layer, #drop7_layer,
                 fc8_layer]

backend = DefaultBackend()
init(backend)

net = Net("CIFAR10-train", backend, [data_tr_layer, common_layers..., loss_layer])

lr_policy = LRPolicy.Staged(
  (60000, LRPolicy.Fixed(0.001)),
  (5000, LRPolicy.Fixed(0.0001)),
  (5000, LRPolicy.Fixed(0.00001)),
)
method = SGD()
solver_params = make_solver_parameters(method, max_iter=70000,
    regu_coef=0.004, mom_policy=MomPolicy.Fixed(0.9), lr_policy=lr_policy,
    load_from="snapshots")
solver = Solver(method, solver_params)

# report training progress every 200 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=200)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot("snapshots"), every_n_iter=5000)

# show performance on test data every 1000 iterations
test_net = Net("CIFAR10-test", backend, [data_tt_layer, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)
