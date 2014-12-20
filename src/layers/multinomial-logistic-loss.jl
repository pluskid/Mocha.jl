############################################################
# Multinomial Logistic Loss
############################################################
@defstruct MultinomialLogisticLossLayer Layer (
  name :: String = "multinomial-logistic-loss",
  weights :: Array = [],
  (dim :: Int = -2, dim != 0),
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

  op_dim       :: Int
  weights_blob :: Blob
end

function setup(backend::Backend, layer::MultinomialLogisticLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  tensor_dim = ndims(inputs[1])
  dims = size(inputs[1])

  op_dim = layer.dim
  if op_dim < 0
    op_dim += tensor_dim + 1
  end
  @assert 1 <= op_dim <= tensor_dim
  @assert op_dim != tensor_dim # the last dimension is the mini-batch dimension

  # weights for each class
  if isempty(layer.weights)
    weights_blob = NullBlob()
  else
    @assert op_dim == tensor_dim-1 "When weights provided, LogisticLoss can only operate on the second-to-last dimension"
    weights = layer.weights
    if ndims(weights) == 1
      if length(weights) != dims[op_dim]
        error("Invalid weights: size should be equal to number of classes")
      end
      new_shape = ones(Int, tensor_dim-1); new_shape[op_dim] = dims[op_dim]
      rep_shape = [dims[1:end-1]...]; rep_shape[op_dim] = 1
      weights = repeat(reshape(weights, new_shape...), inner=rep_shape)
    end
    if ndims(weights) != tensor_dim-1 || size(weights) != dims[1:end-1]
      error("Invalid weights: should be either a ND-tensor of one data point or a vector of (classes)")
    end
    weights = convert(Array{data_type}, weights)

    if layer.normalize == :local
      weights = weights .* (dims[op_dim] ./ sum(weights, op_dim))
    elseif layer.normalize == :global
      weights = weights * (prod(size(weights)) / sum(weights))
    else
      @assert layer.normalize == :no
    end

    weights_blob = make_blob(backend, weights)
  end

  state = MultinomialLogisticLossLayerState(layer, convert(data_type, 0), op_dim, weights_blob)
  return state
end
function shutdown(backend::Backend, state::MultinomialLogisticLossLayerState)
end

function forward(backend::CPUBackend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data

  dims = size(pred)
  label_dim = [i == state.op_dim ? 1 : dims[i] for i = 1:length(dims)]
  label = reshape(label, label_dim...)

  idx_all = map(1:length(dims)) do i
    if i == state.op_dim
      int(label) + 1
    else
      dim = dims[i]
      reshape(1:dim, [j == i? dim : 1 for j = 1:length(dims)]...)
    end
  end

  if isa(state.weights_blob, NullBlob)
    loss = sum(-log(max(broadcast_getindex(pred, idx_all...), 1e-20)))
  else
    tmp = reshape([1], ones(Int, length(dims))...)
    loss = sum(-log(max(broadcast_getindex(pred, idx_all...), 1e-20)) .*
        broadcast_getindex(state.weights_blob.data, idx_all[1:end-1]..., tmp))
  end
  state.loss = loss / (prod(dims) / dims[state.op_dim])
end

function backward(backend::Backend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

