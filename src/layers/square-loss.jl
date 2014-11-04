############################################################
# Square Loss
#
# L(\hat{y},y) = 1/2N \sum_{i=1}^N (\hat{y}_i - y_i)^2
############################################################
@defstruct SquareLossLayer LossLayer (
  (bottoms :: Vector{String} = String[], length(bottoms) == 2),
)

type SquareLossLayerState{T} <: LayerState
  layer :: SquareLossLayer
  loss  :: T

  # a helper blob used to compute the loss without destroying
  # the pred results passed up
  pred_copy :: Blob
end

function setup(sys::System, layer::SquareLossLayer, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  if isa(sys.backend, CPUBackend)
    pred_copy = CPUBlob(data_type, size(inputs[1])...)
  elseif isa(sys.backend, CuDNNBackend)
    pred_copy = cudnn_make_tensor_blob(data_type, size(inputs[1])...)
  else
    error("Backend $(sys.backend) not supported")
  end

  state = SquareLossLayerState(layer, convert(data_type, 0), pred_copy)
  return state
end

function forward(sys::System{CPUBackend}, state::SquareLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  BLAS.axpy!(n, convert(data_type, -1), label.data, 1, state.pred_copy.data, 1)
  state.loss = 0.5/get_num(pred)*BLAS.dot(state.pred_copy.data, state.pred_copy.data)
end

function backward(sys::System{CPUBackend}, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
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


function forward(sys::System{CuDNNBackend}, state::SquareLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, -1), label.ptr, 1, state.pred_copy.ptr, 1)
  state.loss = 0.5/get_num(pred)*CuBLAS.dot(sys.backend.cublas_ctx, n, state.pred_copy.ptr, 1, state.pred_copy.ptr, 1)
end

function backward(sys::System{CuDNNBackend}, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    pred = inputs[1]
    label = inputs[2]

    data_type = eltype(pred)
    n = length(pred)
    num = get_num(pred)

    erase!(diff)
    CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, 1.0/num), pred.ptr, 1, diff.ptr, 1)
    CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, -1.0/num), label.ptr, 1, diff.ptr, 1)
  end
end

