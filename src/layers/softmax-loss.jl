############################################################
# Softmax Loss
############################################################
@defstruct SoftmaxLossLayer LossLayer (
  name :: String = "softmax-loss",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)

type SoftmaxLossLayerState{T} <: LayerState
  layer    :: SoftmaxLossLayer
  loss     :: T

  softmax  :: SoftmaxLayerState
  logistic :: MultinomialLogisticLossLayerState
end

function setup(sys::System, layer::SoftmaxLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])

  softmax_layer = SoftmaxLayer(tops=Array(Symbol, length(inputs)), bottoms=Array(Symbol, length(inputs)))
  softmax = setup(sys, softmax_layer, Blob[inputs[1]], Blob[])

  logistic_layer = MultinomialLogisticLossLayer(bottoms=Array(Symbol, 2))
  logistic = setup(sys, logistic_layer, inputs, Blob[])

  state = SoftmaxLossLayerState(layer, convert(data_type, 0), softmax, logistic)
  return state
end

function prepare_backward(sys::System, state::SoftmaxLossLayerState)
end

function backward(sys::System{CPUBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CPUBlob)
    copy!(diff, state.softmax.blobs[1])
    width, height, channels, num = size(diff)

    index = (reshape(1:width, (width, 1, 1, 1)),
        reshape(1:height, (1, height, 1, 1)),
        int(inputs[2].data)+1, reshape(1:num, (1, 1, 1, num)))
    broadcast_setindex!(diff.data, broadcast_getindex(diff.data, index...)-1, index...)
    diff.data /= width*height*num
  end
end

function forward(sys::System, state::SoftmaxLossLayerState, inputs::Vector{Blob})
  forward(sys, state.softmax, Blob[inputs[1]])
  forward(sys, state.logistic, Blob[state.softmax.blobs[1], inputs[2]])
  state.loss = state.logistic.loss
end

function shutdown(sys::System, state::SoftmaxLossLayerState)
  shutdown(sys, state.softmax)
  shutdown(sys, state.logistic)
end

