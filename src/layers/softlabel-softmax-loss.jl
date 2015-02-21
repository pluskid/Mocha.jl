@defstruct SoftlabelSoftmaxLossLayer Layer (
  name :: String = "softlabel-softmax-loss",
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

  softmax_loss :: SoftmaxLossLayerState
  fake_labels :: Vector{Blob}
  fake_diff   :: Blob
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

  # weights will be the soft labels
  weights = zeros(data_type, size(inputs[1]))
  softmax_loss_layer = SoftmaxLossLayer(bottoms=Array(Symbol, length(inputs)), dim=layer.dim,
      weights=weights, normalize=:no)

  dims = size(inputs[1])
  dims_label = map(1:length(dims)) do i
    i == op_dim ? 1 : dims[i]
  end
  fake_labels = Blob[make_blob(backend, zeros(data_type, dims_label...)+(i-1)) for i = 1:dims[op_dim]]
  fake_diff = make_blob(backend, data_type, dims)

  softmax_loss = setup(backend, softmax_loss_layer, Blob[inputs[1], fake_labels[1]], diffs)
  state = SoftlabelSoftmaxLossLayerState(layer, convert(data_type, 0), softmax_loss, fake_labels, fake_diff)
  return state
end

function shutdown(backend::Backend, state::SoftlabelSoftmaxLossLayerState)
  shutdown(backend, state.softmax_loss)
  map(destroy, state.fake_labels)
end

function forward(backend::Backend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  state.loss = 0

  copy!(state.softmax_loss.logistic.weights_blob, label)
  for i = 1:length(state.fake_labels)
    forward(backend, state.softmax_loss, Blob[pred, state.fake_labels[i]])
    state.loss += state.softmax_loss.loss
  end
end

function backward(backend::CPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CPUBlob)
    pred = inputs[1]
    label = inputs[2]
    data_type = eltype(pred)
    copy!(state.softmax_loss.logistic.weights_blob, label)
    erase!(diff)

    for i = 1:length(state.fake_labels)
      backward(backend, state.softmax_loss, Blob[pred, state.fake_labels[i]], Blob[state.fake_diff])
      BLAS.axpy!(length(pred), one(data_type), state.fake_diff.data, 1, diff.data, 1)
    end
  end
end
