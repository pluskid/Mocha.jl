############################################################
# Softmax Loss
############################################################
@defstruct SoftmaxLossLayer LossLayer (
  (bottoms :: Vector{String} = String[], length(bottoms) == 2),
)

type SoftmaxLossLayerState{T} <: LayerState
  layer    :: SoftmaxLossLayer
  loss     :: T

  softmax  :: SoftmaxLayerState
  logistic :: MultinomialLogisticLossLayerState

  etc      :: Any
end

function setup(sys::System, layer::SoftmaxLossLayer, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  etc = nothing

  softmax_layer = SoftmaxLayer(tops=Array(String, length(inputs)), bottoms=Array(String, length(inputs)))
  softmax = setup(sys, softmax_layer, Blob[inputs[1]])

  logistic_layer = MultinomialLogisticLossLayer(bottoms=Array(String, 2))
  logistic = setup(sys, logistic_layer, inputs)

  state = SoftmaxLossLayerState(layer, convert(data_type, 0), softmax, logistic, etc)
  return state
end

function forward(sys::System{CPUBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  # substract max before exp to avoid numerical issue
  # also do in-place computation to get the probability so that we do not need to
  # do duplicated computation in backward pass when computing the gradient
  pred.data .-= max(pred.data,2)
  pred.data = exp(pred.data)
  pred.data ./= sum(pred.data,2)

  loss = sum(-log(broadcast_getindex(pred.data, vec(1:length(label.data)), vec(int(label.data)))))
  loss /= length(label.data)
  state.blobs[1].data[:] = loss
end

function backward(sys::System{CPUBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  prob  = inputs[1].data
  label = int(inputs[2].data)

  for i = 1:length(label)
    prob[i,label[i]] -= 1
  end
  diffs[1].data[:] = prob[:]
end

function forward(sys::System{CuDNNBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob})
  forward(sys, state.softmax, Blob[inputs[1]])
  forward(sys, state.logistic, Blob[state.softmax.blobs[1], inputs[2]])
  state.loss = state.logistic.loss
end

function backward(sys::System{CuDNNBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    copy!(diff, state.softmax.blobs[1])

    data_type = eltype(diff)
    height, width, channels, num = size(diff)

    spatial_dim = height*width
    prob_dim = channels

    x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X))
    y_block = spatial_dim

    if data_type == Float32
      kernel = sys.backend.mocha.softmax_loss_backward_float
    elseif data_type == Float64
      kernel = sys.backend.mocha.softmax_loss_backward_double
    else
      error("Unsupported data type $data_type")
    end
    CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
        (diff.ptr.p, inputs[2].ptr.p, num, spatial_dim, prob_dim))
    CuBLAS.scal(sys.backend.cublas_ctx, length(diff), convert(data_type, 1.0/(spatial_dim*num)),
        diff.ptr, 1)
  end
end
