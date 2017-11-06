@defstruct SoftlabelSoftmaxLossLayer Layer (
  name :: AbstractString = "softlabel-softmax-loss",
  (weight :: AbstractFloat = 1.0, weight >= 0),
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(SoftlabelSoftmaxLossLayer,
  has_loss => true,
  can_do_bp => true,
  is_sink => true
)

type SoftlabelSoftmaxLossLayerState{T} <: LayerState
  layer :: SoftlabelSoftmaxLossLayer
  loss  :: T

  softmax :: SoftmaxLayerState
  op_dim  :: Int
  etc     :: Any
end

function setup_etc(backend::CPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
end
function setup(backend::Backend, layer::SoftlabelSoftmaxLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  tensor_dim = ndims(inputs[1])
  op_dim = layer.dim
  if op_dim < 0
    op_dim += tensor_dim + 1
  end
  @assert 1 <= op_dim <= tensor_dim
  @assert op_dim != tensor_dim # the last dimension is the mini-batch dimension

  softmax_layer = SoftmaxLayer(bottoms=[:softmax_input], tops=[:softmax_output], dim=layer.dim)
  softmax = setup(backend, softmax_layer, Blob[inputs[1]], Blob[diffs[1]])
  state = SoftlabelSoftmaxLossLayerState(layer, zero(data_type), softmax, op_dim, nothing)

  setup_etc(backend, state, inputs)

  return state
end

function shutdown_etc(backend::CPUBackend, state::SoftlabelSoftmaxLossLayerState)
end
function shutdown(backend::Backend, state::SoftlabelSoftmaxLossLayerState)
  shutdown_etc(backend, state)
  shutdown(backend, state.softmax)
end

function forward(backend::CPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  forward(backend, state.softmax, Blob[pred])
  prob = state.softmax.blobs[1]

  dims = size(prob)
  state.loss = state.layer.weight * sum(-vec(log.(max.(prob.data, 1e-20))) .* vec(label.data)) / (prod(dims) / dims[state.op_dim])
end

function backward(backend::CPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CPUBlob)
    diff  = diff.data
    pred  = inputs[1].data
    label = inputs[2].data

    data_type = eltype(pred)
    copy!(diff, state.softmax.blobs[1])
    BLAS.axpy!(length(pred), -one(data_type), label, 1, diff, 1)
    dims = size(pred)
    Vec.mul_scal!(diff, state.layer.weight * dims[state.op_dim]/prod(dims))
  end
end
