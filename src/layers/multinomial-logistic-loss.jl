############################################################
# Multinomial Logistic Loss
############################################################
@defstruct MultinomialLogisticLossLayer LossLayer (
  name :: String = "multinomial-logistic-loss",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)

type MultinomialLogisticLossLayerState{T} <: LayerState
  layer :: MultinomialLogisticLossLayer
  loss  :: T
end

function setup(sys::System, layer::MultinomialLogisticLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  state = MultinomialLogisticLossLayerState(layer, convert(data_type, 0))
  return state
end

function forward(sys::System{CPUBackend}, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  width, height, channels, num = size(pred)

  loss = sum(-log(max(broadcast_getindex(pred, reshape(1:width, (width, 1, 1, 1)),
      reshape(1:height, (1, height, 1, 1)),
      int(label)+1, reshape(1:num, (1, 1, 1, num))), 1e-20)))
  state.loss = loss / (width*height*num)
end

function prepare_backward(sys::System, state::MultinomialLogisticLossLayerState)
end

function backward(sys::System, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

function shutdown(sys::System, state::MultinomialLogisticLossLayerState)
end

