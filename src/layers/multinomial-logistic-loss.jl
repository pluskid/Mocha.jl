############################################################
# Multinomial Logistic Loss
############################################################
@defstruct MultinomialLogisticLossLayer Layer (
  name :: String = "multinomial-logistic-loss",
  weights :: Array = [],
  (normalize:: Symbol = :local, in(normalize,[:local,:global,:no])),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(MultinomialLogisticLossLayer,
  has_loss => true,
  is_sink  => true,
)

type MultinomialLogisticLossLayerState{T} <: LayerState
  layer :: MultinomialLogisticLossLayer
  loss  :: T

  weights_blob :: Blob
end

function setup(backend::Backend, layer::MultinomialLogisticLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  width, height, channels, num = size(inputs[1])

  # weights for each class
  if isempty(layer.weights)
    weights_blob = NullBlob()
  else
    weights = layer.weights
    if ndims(weights) == 1
      if length(weights) != channels
        error("Invalid weights: size should be equal to number of classes")
      end
      weights = repeat(reshape(weights,1,1,channels), inner=[width,height,1])
    end
    if ndims(weights) != 3 || size(weights) != (width,height,channels)
      error("Invalid weights: should be either a 3-tensor of (width,height,channels) or a vector of (channels)")
    end
    weights = convert(Array{data_type}, weights)

    if layer.normalize == :local
      weights = weights .* (channels ./ sum(weights, 3))
    elseif layer.normalize == :global
      weights = weights * (width*height*channels / sum(weights))
    else
      @assert layer.normalize == :no
    end

    weights_blob = make_blob(backend, reshape(weights, width,height,channels,1))
  end

  state = MultinomialLogisticLossLayerState(layer, convert(data_type, 0), weights_blob)
  return state
end
function shutdown(backend::Backend, state::MultinomialLogisticLossLayerState)
end

function forward(backend::CPUBackend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  width, height, channels, num = size(pred)

  idx_width  = reshape(1:width, (width, 1, 1, 1))
  idx_height = reshape(1:height, (1, height, 1, 1))
  idx_chann  = int(label)+1
  idx_num    = reshape(1:num, (1, 1, 1, num))

  if isa(state.weights_blob, NullBlob)
    loss = sum(-log(max(broadcast_getindex(pred, idx_width, idx_height, idx_chann, idx_num), 1e-20)))
  else
    loss = sum(-log(max(broadcast_getindex(pred, idx_width, idx_height, idx_chann, idx_num), 1e-20)) .*
        broadcast_getindex(state.weights_blob.data, idx_width, idx_height, idx_chann, reshape([1],1,1,1,1)))
  end
  state.loss = loss / (width*height*num)
end

function backward(backend::Backend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

