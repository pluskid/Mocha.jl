module mnistVAE
#ENV["MOCHA_USE_NATIVE_EXT"] = "true"
#ENV["OMP_NUM_THREADS"] = 1
#blas_set_num_threads(1)
ENV["MOCHA_USE_CUDA"] = "true"
## if haskey(ENV, "MOCHA_USE_CUDA")
##     pop!(ENV, "MOCHA_USE_CUDA")
## end

using Mocha

############################################################
# Based on the paper "Auto-Encoding Variational Bayes"
# by Diederik P Kingma, Max Welling
# http://arxiv.org/abs/1312.6114
#
# We perform unsupervised learning of a generative model of
# binary MNIST data, using the "Variational Auto-Encoder" architecture.
#
# 50-dim Gaussian latent variable z, with a standard normal prior
#
# Encoder MLP:
# - one hidden layer with 500 hidden units and tanh()
# - two linear readout layers for mu(z) and logvar(z)
#
# (We'll just use elementwise layers and the RandomNormal layer to
# implement the "reparameterization trick".)
#
# Decoder MLP matches Encoder:
# - one hidden layer with 500 hidden units and tanh()
# - one linear readout layer for p(x)
#
# Minibatches with batch size 100 (SGD with Momentum, though paper uses AdaGrad)
############################################################


# fix the random seed to make results reproducable
srand(12345678) # TODO check this goes through GPU

const N_LATENT = 50
const N_HIDDEN_ENC = 500
const N_HIDDEN_DEC = 500
const N_OUT = 784


data_layer      = HDF5DataLayer(name="train-data", source="data/train.txt", batch_size=100)

enc1_layer      = InnerProductLayer(name="enc1", output_dim=N_HIDDEN_ENC, neuron=Neurons.Tanh(),
                                    weight_init = GaussianInitializer(std=0.01),
                                    bottoms=[:data], tops=[:enc1])
enc_split_layer = SplitLayer(name="enc-split", bottoms=[:enc1], tops=[:enc1_mu_in, :enc1_sigma_in])

enc_mu_layer    = InnerProductLayer(name="enc1-mu", output_dim=N_LATENT, neuron=Neurons.Identity(),
                                    weight_init = GaussianInitializer(std=0.01),
                                    bottoms=[:enc1_mu_in], tops=[:z_mu])
enc_sigma_layer = InnerProductLayer(name="enc1-sigma", output_dim=N_LATENT, neuron=Neurons.ReLU(), # This layer outputs log(sigma^2) in the original paper
                                    weight_init = GaussianInitializer(std=0.01),
                                    bottoms=[:enc1_sigma_in], tops=[:z_sigma])
zm_split_layer = SplitLayer(name="zm-split", bottoms=[:z_mu],
                            tops=[:z_mu_1, :z_mu_2])
zs_split_layer = SplitLayer(name="zs-split", bottoms=[:z_sigma],
                            tops=[:z_sigma_1, :z_sigma_2])

eps_layer       = RandomNormalLayer(name="eps",
                                    batch_sizes=[data_layer.batch_size], output_dims=[N_LATENT],
                                    tops=[:eps])
z_noise_layer   = ElementWiseLayer(name="z-noise", bottoms=[:z_sigma_1, :eps], tops=[:z_noise],
                                   operation=ElementWiseFunctors.Multiply())
z_layer         = ElementWiseLayer(name="z", bottoms=[:z_noise, :z_mu_1], tops=[:z],
                                   operation=ElementWiseFunctors.Add())

dec1_layer      = InnerProductLayer(name="dec1", output_dim=N_HIDDEN_DEC, neuron=Neurons.Tanh(),
                                    weight_init = GaussianInitializer(std=0.01),
                                    bottoms=[:z], tops=[:dec1])

dec_out_layer   = InnerProductLayer(name="dec-out", output_dim=N_OUT, neuron=Neurons.Sigmoid(),
                                    weight_init = GaussianInitializer(std=0.01),
                                    bottoms=[:dec1], tops=[:out])

dec_split_layer = SplitLayer(name="dec-split", bottoms=[:out], tops=[:dec_out1, :dec_out2])


re_out_layer    = ConcatLayer(name="reshape-out", bottoms=[:dec_out1, :dec_out2], tops=[:out_reshape])
#re_out_layer    = ReshapeLayer(name="reshape-out", shape=(784,1), bottoms=[:out], tops=[:out_reshape])

re_data_layer   = ReshapeLayer(name="reshape-data", shape=(784,1), bottoms=[:data], tops=[:data_reshape])
loss_layer      = SoftmaxLossLayer(name="loss", bottoms=[:out_reshape, :data_reshape])

kl_loss_layer   = GaussianKLLossLayer(name="kl-loss", bottoms=[:z_mu_2, :z_sigma_2])

#backend = GPUBackend()
backend = CPUBackend()
init(backend)

enc_layers = [enc1_layer, enc_split_layer, enc_mu_layer, enc_sigma_layer, zm_split_layer, zs_split_layer]
z_layers = [eps_layer, z_noise_layer, z_layer]
dec_layers = [dec1_layer, dec_out_layer]
common_layers = [enc_layers, z_layers, dec_layers]
loss_layers = [dec_split_layer, re_out_layer, re_data_layer, loss_layer, kl_loss_layer]
# put training net together, note that the correct ordering will automatically be established by the constructor
net = Net("MNIST-VAE-train", backend, [data_layer, common_layers..., loss_layers...])

base_dir = "snapshots_mnist_VAE"
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
## data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
## acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:out, :label], report_error=true)
## test_net = Net("MNIST-var-test", backend, [data_layer_test, common_layers..., acc_layer])
## add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=600)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
## destroy(test_net)
shutdown(backend)

end
