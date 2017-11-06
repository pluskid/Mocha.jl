############################################################
# BinaryCrossEntropyLossLayer
############################################################
@defstruct BinaryCrossEntropyLossLayer Layer (
  name :: AbstractString = "binary-cross-entropy-loss",
  (weight :: AbstractFloat = 1.0, weight >= 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(BinaryCrossEntropyLossLayer,
  has_loss => true,
  is_sink  => true,
  can_do_bp => true,
)

type BinaryCrossEntropyLossLayerState{T} <: LayerState
  layer :: BinaryCrossEntropyLossLayer
  loss  :: T
end

function setup(backend::Backend, layer::BinaryCrossEntropyLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  state = BinaryCrossEntropyLossLayerState(layer, convert(data_type, 0))
  return state
end

function shutdown(backend::Backend, state::BinaryCrossEntropyLossLayerState)
end

function forward(backend::CPUBackend, state::BinaryCrossEntropyLossLayerState, inputs::Vector{Blob})
  pred = vec(inputs[1].data)
  label = vec(inputs[2].data)
  loss = BLAS.dot(log.(pred), label) + BLAS.dot(log1p.(-pred), (1-label))

  num = get_num(inputs[1])
  state.loss = state.layer.weight * -loss/num
end

function backward(backend::CPUBackend, state::BinaryCrossEntropyLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CPUBlob)

    # Diffs is df/dloss
    # we want to multiply this by dloss/dpred and dl

    label = inputs[2].data
    pred  = reshape(inputs[1].data, size(label))

    # l = -w sum_i log(p_i) y_i + (log(1-p_i)(1-y_i)
    # dl/dp_i = -w ( y_i/p_i - ((1-y_i)(1-p_i) )
    # dl/dy_i = -w ( log(p_i) - (log(1-p_i) ) = -w log(p_i/(1-p_i))

    n = length(pred)
    a = convert(eltype(pred), -state.layer.weight/get_num(inputs[1]))

    erase!(diff) # is this correct? square-loss does it - should we allow for any incoming diff?
    dl_dpred = (label ./ pred) - ((1-label) ./ (1-pred)) # dloss/d?
    BLAS.axpy!(n, a, dl_dpred, 1, diff.data, 1)
  end
  diff = diffs[2]
  if isa(diff, CPUBlob)
    dl_dlabel = log.(pred ./ (1-pred))
    erase!(diff) # is this correct? square-loss does it
    BLAS.axpy!(n, a, dl_dlabel, 1, diff.data, 1)
  end
end
