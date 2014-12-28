################################################################################
# Configuration
################################################################################
ENV["MOCHA_USE_CUDA"] = "true"

n_hidden_layer   = 2
n_hidden_unit    = 1000
neuron           = Neurons.Sigmoid()
param_key_prefix = "ip-layer"
corruption_rates = [0.1,0.2,0.3]
pretrain_epoch   = 15
finetune_epoch   = 1000
batch_size       = 100
momentum         = 0.0
pretrain_lr      = 0.001
finetune_lr      = 0.1

################################################################################
# Construct the Net
################################################################################
using Mocha
srand(12345678)

backend = GPUBackend()
init(backend)

data_layer = HDF5DataLayer(name="train-data", source="data/train.txt",
    batch_size=batch_size, shuffle=@windows ? false : true)
rename_layer = IdentityLayer(bottoms=[:data], tops=[:ip0])

hidden_layers = [
  InnerProductLayer(name="ip-$i", param_key="$param_key_prefix-$i",
      output_dim=n_hidden_unit, neuron=neuron,
      bottoms=[symbol("ip$(i-1)")], tops=[symbol("ip$i")])
  for i = 1:n_hidden_layer
]

pred_layer = InnerProductLayer(name="pred", output_dim=10,
    bottoms=[symbol("ip$n_hidden_layer")], tops=[:pred])
loss_layer = SoftmaxLossLayer(bottoms=[:pred, :label])

net = Net("MNIST", backend, [data_layer, rename_layer, hidden_layers..., pred_layer, loss_layer])

################################################################################
# Layerwise pre-training for hidden layers
################################################################################
for i = 1:n_hidden_layer
  recon_layer = TiedInnerProductLayer(name="tied-ip-$i", tied_param_key="$param_key_prefix-$i",
      tops=[:recon], bottoms=[symbol("ip$i")])
  recon_loss_layer = SquareLossLayer(bottoms=[:recon, :orig_data])
  recon_data_layer = SplitLayer(bottoms=[:data], tops=[:orig_data, :corrupt_data])
  corrupt_layer = RandomMaskLayer(ratio=corruption_rates[i], bottoms=[:corrupt_data])

  da_layers = [data_layer, recon_data_layer, corrupt_layer, hidden_layers[1:i]...,
      recon_layer, recon_loss_layer]
  da = Net("Denoising-Autoencoder-$i", backend, da_layers)

  # freeze all but the layers for auto-encoder
  freeze_all!(net)
  unfreeze!(net, "ip-$i", "tied-ip-$i")

  base_dir = "pretrain-$i"
  pretrain_params  = SolverParameters(max_iter=div(pretrain_epoch*60000,batch_size),
      regu_coef=0.0, mom_policy=MomPolicy.Fixed(momentum),
      lr_policy=LRPolicy.Fixed(pretrain_lr), load_from=base_dir)
  solver = SGD(pretrain_params)

  add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
  add_coffee_break(solver, Snapshot(base_dir), every_n_iter=10000)
  solve(solver, da)

  destroy(da)
end

base_dir = "finetune"
params = SolverParameters(max_iter=div(finetune_epoch*60000,batch_size),
    regu_coef=0.0, mom_policy=MomPolicy.Fixed(momentum),
    lr_policy=LRPolicy.Fixed(finetune_lr), load_from=base_dir)
solver = SGD(params)

add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=10000)

data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:pred, :label])
test_net = Net("MNIST-test", backend, [data_layer_test, rename_layer,
    hidden_layers..., pred_layer, acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=5000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)
