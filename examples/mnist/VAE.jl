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
############################################################

function make_vae(backend, N_LATENT=50, N_HIDDEN_DEC=500, N_HIDDEN_ENC=500, N_OUT=784)

  data_layer      = HDF5DataLayer(name="train-data", source="data/train.txt", batch_size=100)

  enc1_layer      = InnerProductLayer(name="enc1", output_dim=N_HIDDEN_ENC, neuron=Neurons.Tanh(),
                                      weight_init = GaussianInitializer(std=0.01),
                                      bottoms=[:data], tops=[:enc1])
  enc_split_layer = SplitLayer(name="enc-split", bottoms=[:enc1], tops=[:enc1_mu_in, :enc1_sigma_in])

  enc_mu_layer    = InnerProductLayer(name="enc1-mu", output_dim=N_LATENT, neuron=Neurons.Identity(),
                                      weight_init = GaussianInitializer(std=0.01),
                                      bottoms=[:enc1_mu_in], tops=[:z_mu])
  enc_sigma_layer = InnerProductLayer(name="enc1-sigma", output_dim=N_LATENT,
                                      # This layer outputs log(sigma^2) in the original paper
                                      neuron=Neurons.Exponential(), # still has large jumps in KL on CPU
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


  bce_loss_layer  = BinaryCrossEntropyLossLayer(name="bce-loss", bottoms=[:out, :data])

  kl_loss_layer   = GaussianKLLossLayer(name="kl-loss", bottoms=[:z_mu_2, :z_sigma_2])

  enc_layers = [enc1_layer, enc_split_layer, enc_mu_layer, enc_sigma_layer, zm_split_layer, zs_split_layer]
  z_layers = [eps_layer, z_noise_layer, z_layer]
  dec_layers = [dec1_layer, dec_out_layer]
  common_layers = [enc_layers, z_layers, dec_layers]
  loss_layers = [bce_loss_layer, kl_loss_layer]
  # put training net together, note that the correct ordering will automatically be established by the constructor
  net = Net("MNIST-VAE-train", backend, [data_layer, common_layers..., loss_layers...])
  return net
end


# This function reproduces the feed-forward part of the model above, layers 12-13,
# as a simple function

asarray{T}(blob::CuTensorBlob{T}) = begin
  arr = zeros(T, blob.shape)
  copy!(arr, blob)
  return arr
end

asarray(blob::CPUBlob) = blob.data

function make_latent_to_output(net::Net, dec1_index, dec_out_index)
  sigmoid(x) = 1 ./ (1 + exp(-x))

  function latent_to_output(z)
    # pad z with zeros to match the expected latent size
    N_Z = size(net.states[dec1_index].parameters[1].blob)[1]
    if length(z) < N_Z
      z_ = zeros(eltype(z), N_Z)
      z_[1:length(z)] = z
      z = z_
    end

    w1 = asarray(net.states[dec1_index].parameters[1].blob)
    b1 = asarray(net.states[dec1_index].parameters[2].blob)

    w2 = asarray(net.states[dec_out_index].parameters[1].blob)
    b2 = asarray(net.states[dec_out_index].parameters[2].blob)


    dec1 = tanh(w1' * z + b1)
    dec_out = sigmoid(w2' * dec1 + b2)
    return reshape(dec_out, 28, 28)'
  end
  return latent_to_output
end
