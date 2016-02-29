type CuMultinomialLogisticLossEtcState{T}
  loss_blob     :: CuTensorBlob{T}
  loss          :: Array{T}
  spatial_dim   :: Int
  num           :: Int
end

function setup_etc(backend::GPUBackend, layer::MultinomialLogisticLossLayer, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  loss_blob = make_zero_blob(backend, data_type, 1, 1, 1, 1)
  loss = data_type[0]
  return CuMultinomialLogisticLossEtcState(loss_blob, loss, 0, 0)
end

function shutdown(backend::GPUBackend, state::MultinomialLogisticLossLayerState)
  custate = state.etc
  destroy(custate.loss_blob)
end

function forward(backend::GPUBackend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred      = inputs[1]
  label     = inputs[2]
  data_type = eltype(pred)
  custate = state.etc

  spatial_dim, channels, num = split_dims(pred, state.op_dim)
  prob_dim = channels

  x_block = round(Int, ceil(convert(Float64, num)/CUDA.THREADS_PER_BLOCK_X))
  y_block = spatial_dim

  if data_type == Float32
    kernel = backend.mocha.logistic_loss_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.logistic_loss_forward_double
  else
    error("Unsupported data type $data_type")
  end

  if isa(state.weights_blob, NullBlob)
    weights = convert(Ptr{data_type}, 0)
  else
    weights = get_ptr(state.weights_blob).p
  end

  CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
      (get_ptr(pred).p, get_ptr(label).p, weights, num, spatial_dim, prob_dim, get_ptr(custate.loss_blob).p))

  custate.spatial_dim = spatial_dim
  custate.num = num
end

function sync(backend::GPUBackend, state::MultinomialLogisticLossLayerState)
  custate = state.etc
  # use copy to sync
  copy!(custate.loss, custate.loss_blob)
  erase!(custate.loss_blob)

  state.loss = state.layer.weight * custate.loss[1] / (custate.spatial_dim * custate.num)
end

