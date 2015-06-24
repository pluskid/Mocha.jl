############################################################
# Gaussian KL loss
#
# Given inputs mu and sigma, describing a diagonal covariance
# Gaussian distribution, the loss incurred is the
# Kullback-Leibler divergence from the input distribution
# to the standard Gaussian N(0,I).
############################################################
@defstruct GaussianKLLossLayer Layer (
  name :: String = "gauss-kl-loss",
  (weight :: FloatingPoint = 1.0, weight >= 0),
  (bottoms :: Vector{Symbol} = Symbol[:mu, :sigma], length(bottoms) == 2),
)
@characterize_layer(GaussianKLLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true,
)

type GaussianKLLossLayerState{T} <: LayerState
  layer      :: GaussianKLLossLayer
  loss       :: T
end

function setup(backend::Backend, layer::GaussianKLLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])

  state = GaussianKLLossLayerState(layer, zero(data_type))
  return state
end
function shutdown(backend::Backend, state::GaussianKLLossLayerState)
    # nothing
end

const log2π = log(2π)


function forward(backend::CPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob})
  mu  = inputs[1]
  sigma = inputs[2]

  data_type = eltype(mu)
  n = length(mu) # length or num?
  num = get_num(mu)
  #@assert length(sigma) == n
  state.loss = -0.5(n * log2π  + sum(mu.data.^2 + sigma.data.^2)) * -state.layer.weight / num
#    @info("KL fwd: got μ=$(mu.data), σ=$(sigma.data), loss = $(state.loss)")
end

function backward(backend::CPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  num = get_num(inputs[1])
  diff = diffs[1]
  #diff = df/dmu[i]

  if isa(diff, CPUBlob)
    mu    = inputs[1].data
    copy!(diff, mu*state.layer.weight/num)
  end

  diff = diffs[2]
  #diff = df/dsigma[i]
  if isa(diff, CPUBlob)
    sigma = inputs[2].data
    copy!(diff, sigma*state.layer.weight/num)
  end

end
