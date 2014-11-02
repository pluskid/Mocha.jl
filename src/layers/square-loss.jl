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
    # TODO
  elseif isa(sys.backend, CuDNNBackend)
    pred_copy = cudnn_make_pod_blob(data_type, size(inputs[1])...)
  else
    error("Backend $(sys.backend) not supported")
  end

  state = SquareLossLayerState(layer, convert(data_type, 0), pred_copy)
  return state
end

function forward(sys::System{CPUBackend}, state::SquareLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1].data
  label = inputs[2].data

  dims = size(pred)
  batch_size = dims[1]
  rest_dim = prod(dims[2:end])

  pred  = reshape(pred, (batch_size, rest_dim))
  label = reshape(label, (batch_size, rest_dim))

  state.blobs[1].data[:] = 0.5*mean(sum((pred - label).^2, 2))
end

function backward(sys::System{CPUBackend}, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if isa(diffs[1], CPUBlob)
    pred  = inputs[1].data
    label = inputs[2].data
    diffs[1].data[:] = (pred - label) / size(pred,1)
  end
end


function forward(sys::System{CuDNNBackend}, state::SquareLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, -1), label.ptr, 1, state.pred_copy.ptr, 1)
  state.loss = 0.5/size(pred,4)*CuBLAS.dot(sys.backend.cublas_ctx, n, state.pred_copy.ptr, 1, state.pred_copy.ptr, 1)
end

function backward(sys::System{CuDNNBackend}, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    pred = inputs[1]
    label = inputs[2]

    data_type = eltype(pred)
    n = length(pred)
    num = size(pred, 4)

    erase!(diff)
    CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, 1.0/num), pred.ptr, 1, diff.ptr, 1)
    CuBLAS.axpy(sys.backend.cublas_ctx, n, convert(data_type, -1.0/num), label.ptr, 1, diff.ptr, 1)
  end
end

