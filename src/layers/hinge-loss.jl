#########################################################
# Hinge Loss
#
# L(\hat{y},y) = 1/2N \sum_{i=1}^N (1-y_i \hat{y}_i)
#########################################################
@defstruct HingeLossLayer Layer (
    name :: String = "hinge-loss",
    (weight :: FloatingPoint = 1.0, weight >= 0),
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)

@characterize_layer(HingeLossLayer,
    has_loss  => true,
    can_do_bp => true,
    is_sink   => true,
    has_stats => true,
)

type HingeLossLayerState{T} <: LayerState
    layer :: HingeLossLayer
    loss  :: T

    loss_accum :: T
    n_accum    :: Int

#     # a helper blob used to compute the loss without destroying
#     # the prediction results passed up
#     pred_copy :: Blob
#     bp_mask :: Blob

    # a helper blob used to compute the loss
    loss_blob :: Blob
end

function setup(backend::Backend, layer::HingeLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
#     pred_copy = make_blob(backend, data_type, size(inputs[1])...)
#     bp_mask = make_blob(backend, Bool, size(inputs[1])...)
    loss_blob = make_blob(backend, data_type, 1)

    state = HingeLossLayerState(layer, zero(data_type), zero(data_type), 0, loss_blob)#, pred_copy, bp_mask)
    return state
end
function shutdown(backend::Backend, state::HingeLossLayerState)
#     destroy(state.pred_copy)
#     destroy(state.bp_mask)
  destroy(state.loss_blob)
  nothing
end
function reset_statistics(state::HingeLossLayerState)
    stat.n_accum = 0
    state.loss_accum = zero(typeof(state.loss_accum))
end
function dump_statistics(storage, state::HingeLossLayerState, show::Bool)
    update_statistics(storage, "$(state.layer.name)-hinge-loss", state.loss_accum)

    if show
        loss = @sprintf("%.4f", state.loss_accum)
        @info("  Hinge-loss (avg over $(state.n_accum)) = $loss")
    end
end

function forward(backend::CPUBackend, state::HingeLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  state.loss = zero(data_type)
  for i=1:n
    loss = one(data_type) - pred.data[i]*label.data[i]
    if loss > zero(data_type)
      state.loss += loss
    end
  end
  state.loss /= convert(data_type, get_num(pred))

  # accumulate statistics
  state.loss_accum = (state.loss_accum*state.n_accum + state.loss*get_num(pred)) / (state.n_accum+get_num(pred))
  state.n_accum += get_num(pred)
end

function backward(backend::CPUBackend, state::HingeLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  const pred = inputs[1]
  const label = inputs[2]

  const data_type = eltype(pred)
  const n = length(pred)
  const num = get_num(pred)

  if isa(diffs[1], CPUBlob)
    erase!(diffs[1])
    for i = 1:n
      if pred.data[i]*label.data[i] < one(data_type)
        diffs[1].data[i] = convert(data_type, -state.layer.weight/num) * label.data[i]
        end
    end
  end

  if isa(diffs[2], CPUBlob)
    erase!(diffs[2])
    for i = 1:n
      if pred.data[i]*label.data[i] < one(data_type)
        diffs[2].data[i] = convert(data_type, -state.layer.weight/num) * pred.data[i]
        end
    end
  end

end
