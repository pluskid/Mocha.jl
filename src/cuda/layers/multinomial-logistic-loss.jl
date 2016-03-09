#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
type CuMultinomialLogisticLossEtcState{T}
  loss          :: SyncMem{T}
  spatial_dim   :: Int
  num           :: Int
end

function setup_etc(backend::GPUBackend, layer::MultinomialLogisticLossLayer, inputs::Vector{Blob})
  dev_blob = make_zero_blob(backend, Float32, 1, 1, 1, 1)
  loss = SyncMem(backend, dev_blob)
  return CuMultinomialLogisticLossEtcState(loss, 0, 0)
end

function shutdown(backend::GPUBackend, state::MultinomialLogisticLossLayerState)
  custate = state.etc
  destroy(custate.loss)
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
    kernel = get_mocha(backend).logistic_loss_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).logistic_loss_forward_double
  else
    error("Unsupported data type $data_type")
  end

  if isa(state.weights_blob, NullBlob)
    weights = convert(Ptr{data_type}, 0)
  else
    weights = get_ptr(state.weights_blob).p
  end

  CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
      (get_ptr(pred).p, get_ptr(label).p, weights, num, spatial_dim, prob_dim, get_ptr(custate.loss.dev_blob).p), get_stream(backend))

  custate.spatial_dim = spatial_dim
  custate.num = num
end

function sync(backend::GPUBackend, state::MultinomialLogisticLossLayerState)
  custate = state.etc
  sync_all!(custate.loss)
  erase_all!(custate.loss.dev_blob)
end

function calc_loss(backend::GPUBackend, state::MultinomialLogisticLossLayerState)
  custate = state.etc
  state.loss = state.layer.weight * mean(custate.loss.host_blob)[1] / (custate.spatial_dim * custate.num)
  return state.loss
end

