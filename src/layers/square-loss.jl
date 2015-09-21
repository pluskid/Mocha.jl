############################################################
# Square Loss
#
# L(\hat{y},y) = 1/2N \sum_{i=1}^N (\hat{y}_i - y_i)^2
############################################################
@defstruct SquareLossLayer Layer (
  name :: AbstractString = "square-loss",
  (weight :: AbstractFloat = 1.0, weight >= 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(SquareLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true,
  has_stats => true,
)

type SquareLossLayerState{T} <: LayerState
  layer      :: SquareLossLayer
  loss       :: T

  loss_accum :: T
  n_accum    :: Int

  # a helper blob used to compute the loss without destroying
  # the pred results passed up
  pred_copy :: Blob
end

function setup(backend::Backend, layer::SquareLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  pred_copy = make_blob(backend, data_type, size(inputs[1])...)

  state = SquareLossLayerState(layer, zero(data_type), zero(data_type), 0, pred_copy)
  return state
end
function shutdown(backend::Backend, state::SquareLossLayerState)
  destroy(state.pred_copy)
end

function reset_statistics(state::SquareLossLayerState)
  state.n_accum = 0
  state.loss_accum = zero(typeof(state.loss_accum))
end
function dump_statistics(storage, state::SquareLossLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-square-loss", state.loss_accum)

  if show
    loss = @sprintf("%.4f", state.loss_accum)
    @info("  Square-loss (avg over $(state.n_accum)) = $loss")
  end
end

function forward(backend::CPUBackend, state::SquareLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  BLAS.axpy!(n, convert(data_type, -1), label.data, 1, state.pred_copy.data, 1)
  state.loss = state.layer.weight * 0.5/get_num(pred)*BLAS.dot(state.pred_copy.data, state.pred_copy.data)

  # accumulate statistics
  state.loss_accum = (state.loss_accum*state.n_accum + state.loss*get_num(pred)) / (state.n_accum+get_num(pred))
  state.n_accum += get_num(pred)
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
    BLAS.axpy!(n, convert(data_type, state.layer.weight/num), pred.data, 1, diff.data, 1)
    BLAS.axpy!(n, convert(data_type, -state.layer.weight/num), label.data, 1, diff.data, 1)
  end

  # the "label" also needs gradient
  if isa(diffs[2], CPUBlob)
    copy!(diffs[2], diff)
    BLAS.scal!(n, -one(data_type), diffs[2].data, 1)
  end
end

