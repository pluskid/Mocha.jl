############################################################
# Softmax Loss
############################################################
@defstruct SoftmaxLossLayer LossLayer (
  (tops :: Vector{String} = String["softmax-loss"], length(tops) == 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == 2)
)

type SoftmaxLossLayerState <: LayerState
  layer :: SoftmaxLossLayer
  blobs :: Vector{Blob}
end

function setup(sys::System, layer::SoftmaxLossLayer, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  if isa(sys.backend, CPUBackend)
    blobs = Blob[CPUBlob(Array(data_type, 1))]
  else
    error("Backend $(sys.backend) not supported")
  end

  state = SoftmaxLossLayerState(layer, blobs)
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

