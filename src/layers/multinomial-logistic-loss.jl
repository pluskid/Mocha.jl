############################################################
# Multinomial Logistic Loss
############################################################
@defstruct MultinomialLogisticLossLayer Layer (
  name :: String = "multinomial-logistic-loss",
  weights :: Array = [],
  (weight :: FloatingPoint = 1.0, weight >= 0),
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
    weights = layer.weights

    if ndims(weights) == 1
      if length(weights) != dims[op_dim]
        error("Invalid weights: size should be equal to number of classes")
      end
      new_shape = ones(Int, tensor_dim-1); new_shape[op_dim] = dims[op_dim]
      rep_shape = [dims[1:end-1]...]; rep_shape[op_dim] = 1
      weights = repeat(reshape(weights, new_shape...), inner=rep_shape)
    end

    if ndims(weights) == tensor_dim-1
      @assert size(weights) == dims[1:end-1]
      new_shape = [size(weights)..., 1]
      rep_shape = [ones(Int64, length(dims)-1)..., dims[end]]
      weights = repeat(reshape(weights, new_shape...), inner=rep_shape)
    end

    @assert size(weights) == dims
    weights = convert(Array{data_type}, weights)

    if layer.normalize == :local
      weights = weights .* (dims[op_dim] ./ sum(weights, op_dim))
    elseif layer.normalize == :global
      for i = 1:dims[end]
        idx = map(x -> 1:x, dims[1:end-1])
        weights[idx..., i] = weights[idx..., i] * (prod(dims[1:end-1]) / sum(weights[idx..., i]))
      end
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

  # make sure that the labels are in [0..(n_class-1)]
  n_class = size(pred, state.op_dim)
  @assert all(0 .<= label .< n_class) "labels should be index in [0, n_class-1]"

  idx_all = map(1:length(dims)) do i
    if i == state.op_dim
      round(Int64, label) + 1
    else
      dim = dims[i]
      reshape(1:dim, [j == i? dim : 1 for j = 1:length(dims)]...)
    end
  end

  if isa(state.weights_blob, NullBlob)
    loss = sum(-log(max(broadcast_getindex(pred, idx_all...), 1e-20)))
  else
    loss = sum(-log(max(broadcast_getindex(pred, idx_all...), 1e-20)) .*
        broadcast_getindex(state.weights_blob.data, idx_all...))
  end
  state.loss = state.layer.weight * loss / (prod(dims) / dims[state.op_dim])
end

function backward(backend::Backend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

