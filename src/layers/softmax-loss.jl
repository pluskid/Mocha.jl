############################################################
# Softmax Loss
############################################################
@defstruct SoftmaxLossLayer Layer (
  name :: AbstractString = "softmax-loss",
  (weight :: AbstractFloat = 1.0, weight >= 0),
  weights :: Array = [],
  normalize:: Symbol = :local,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(SoftmaxLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true
)

mutable struct SoftmaxLossLayerState{T} <: LayerState
  layer    :: SoftmaxLossLayer
  loss     :: T

  softmax  :: SoftmaxLayerState
  logistic :: MultinomialLogisticLossLayerState
end

function setup(backend::Backend, layer::SoftmaxLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])

  softmax_layer = SoftmaxLayer(tops=Array{Symbol}(undef,length(inputs)), bottoms=Array{Symbol}(undef,length(inputs)), dim=layer.dim)
  softmax = setup(backend, softmax_layer, Blob[inputs[1]], Blob[])

  logistic_layer = MultinomialLogisticLossLayer(bottoms=Array{Symbol}(undef,2),
      weights=layer.weights, normalize=layer.normalize, dim=layer.dim)
  logistic = setup(backend, logistic_layer, inputs, Blob[])

  state = SoftmaxLossLayerState(layer, convert(data_type, 0), softmax, logistic)
  return state
end
function shutdown(backend::Backend, state::SoftmaxLossLayerState)
  shutdown(backend, state.softmax)
  shutdown(backend, state.logistic)
end

function forward(backend::Backend, state::SoftmaxLossLayerState, inputs::Vector{Blob})
  forward(backend, state.softmax, Blob[inputs[1]])
  forward(backend, state.logistic, Blob[state.softmax.blobs[1], inputs[2]])
  state.loss = state.logistic.loss * state.layer.weight
end

function backward(backend::CPUBackend, state::SoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CPUBlob)
    dims = size(diff)
    label_dim = [i == state.logistic.op_dim ? 1 : dims[i] for i = 1:length(dims)]
    label = reshape(inputs[2].data, label_dim...)

    idx_all = map(1:length(dims)) do i
      if i == state.logistic.op_dim
        map(x -> round(Int, x), label) .+ 1
      else
        dim = dims[i]
        reshape(1:dim, [j == i ? dim : 1 for j = 1:length(dims)]...)
      end
    end

    if isa(state.logistic.weights_blob, NullBlob)
      copy!(diff, state.softmax.blobs[1])
    else
      copy!(diff, state.softmax.blobs[1].data .*
            getindex.((state.logistic.weights_blob.data,), idx_all...))
    end

    diff_data = reshape(diff.data, dims)
    if isa(state.logistic.weights_blob, NullBlob)
      setindex!.((diff_data,), getindex.((diff_data,), idx_all...) .- 1, idx_all...)
    else
      setindex!.((diff_data,), getindex.((diff_data,), idx_all...) .-
                 getindex.((state.logistic.weights_blob.data,), idx_all...), idx_all...)
    end
    Vec.mul_scal!(diff.data, state.layer.weight * dims[state.logistic.op_dim]/prod(dims))
  end
end
