############################################################
# Gaussian KL loss
#
# Given inputs mu and sigma, describing a diagonal covariance
# Gaussian distribution, the loss incurred is the
# Kullback-Leibler divergence from the input distribution
# to the standard Gaussian N(0,I).
############################################################

@defstruct GaussianKLLossLayer Layer (
                                      name :: AbstractString = "gauss-kl-loss",
                                      (weight :: AbstractFloat = 1.0, weight >= 0),
                                      (bottoms :: Vector{Symbol} = Symbol[:mu, :sigma], length(bottoms) == 2),
                                      )

@characterize_layer(GaussianKLLossLayer,
                    has_loss  => true,
                    can_do_bp => true,
                    is_sink   => true,
                    has_stats => true,
                    )

type GaussianKLLossLayerState{T, B<:Blob} <: LayerState
  layer      :: GaussianKLLossLayer
  loss       :: T
  loss_accum :: T
  n_accum    :: Int

  tmp_blobs :: Dict{Symbol, B}
end


function setup(backend::CPUBackend, layer::GaussianKLLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  state = GaussianKLLossLayerState(layer, zero(data_type), zero(data_type), 0, Dict{Symbol, CPUBlob}())
  return state
end

function shutdown(backend::Backend, state::GaussianKLLossLayerState)
  for blob in values(state.tmp_blobs)
    destroy(blob)
  end
end

function reset_statistics(state::GaussianKLLossLayerState)
  state.n_accum = 0
  state.loss_accum = zero(typeof(state.loss_accum))
end

function dump_statistics(storage, state::GaussianKLLossLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-encoder-loss", state.loss_accum)

  if show
    loss = @sprintf("%.4f", state.loss_accum)
    @info("  GaussianKL-loss (avg over $(state.n_accum)) = $loss")
  end
end

function forward(backend::CPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  n = get_num(inputs[1])
  nn = length(inputs[1])
  mu = inputs[1].data

  sigma = inputs[2].data

  state.loss = zero(data_type)
  for i in 1:nn
    state.loss += mu[i]^2 + sigma[i]^2 - 2log(sigma[i]) - 1
  end
  state.loss *= 0.5 * state.layer.weight / n

  # accumulate statistics
  state.loss_accum *= state.n_accum
  state.loss_accum += state.loss * n
  state.loss_accum /= state.n_accum + n

  state.n_accum += n
end

function backward(backend::CPUBackend, state::GaussianKLLossLayerState,
                  inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  n = get_num(inputs[1])
  nn = length(inputs[1])
  mu = inputs[1].data
  sigma = inputs[2].data

  if isa(diffs[1], CPUBlob)
    diffs[1].data[:] = mu
    diffs[1].data[:] *= state.layer.weight / n
  end

  if isa(diffs[2], CPUBlob)
    sigma_diffs = diffs[2].data
    sigma_diffs[:] = sigma - (1 ./ sigma)
    diffs[2].data[:] *= state.layer.weight / n
  end
end
