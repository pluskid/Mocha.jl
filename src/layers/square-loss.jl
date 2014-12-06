############################################################
# Square Loss
#
# L(\hat{y},y) = 1/2N \sum_{i=1}^N (\hat{y}_i - y_i)^2
############################################################
@defstruct SquareLossLayer Layer (
  name :: String = "square-loss",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(SquareLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true,
)

type SquareLossLayerState{T} <: LayerState
  layer :: SquareLossLayer
  loss  :: T

  # a helper blob used to compute the loss without destroying
  # the pred results passed up
  pred_copy :: Blob
end

function setup(backend::Backend, layer::SquareLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  pred_copy = make_blob(backend, data_type, size(inputs[1])...)

  state = SquareLossLayerState(layer, convert(data_type, 0), pred_copy)
  return state
end
function shutdown(backend::Backend, state::SquareLossLayerState)
  destroy(state.pred_copy)
end

function forward(backend::CPUBackend, state::SquareLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  BLAS.axpy!(n, convert(data_type, -1), label.data, 1, state.pred_copy.data, 1)
  state.loss = 0.5/get_num(pred)*BLAS.dot(state.pred_copy.data, state.pred_copy.data)
end

function backward(backend::CPUBackend, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CPUBlob)
    pred  = inputs[1]
    label = inputs[2]

    data_type = eltype(pred)
    n = length(pred)
    num = get_num(pred)

    erase!(diff)
    BLAS.axpy!(n, convert(data_type, 1.0/num), pred.data, 1, diff.data, 1)
    BLAS.axpy!(n, convert(data_type, -1.0/num), label.data, 1, diff.data, 1)
  end
end

function backward(backend::Backend, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

