############################################################
# Softmax Layer
############################################################
@defstruct SoftmaxLayer CompLayer (
  tops :: Vector{String} = String[],
  (bottoms :: Vector{String} = String[], length(bottoms) == length(tops))
)

type SoftmaxLayerState <: LayerState
  layer :: SoftmaxLayer
  blobs :: Vector{Blob}
end

function setup(sys::System, layer::SoftmaxLayer, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  blobs = Array(Blob, length(layer.tops))
  if isa(sys.backend, CPU)
    for i = 1:length(blobs)
      blobs[i] = CPUBlob(Array(data_type, size(inputs[i],1), 1))
    end
  else
    error("Backend $(sys.backend) not supported")
  end

  state = SoftmaxLayerState(layer, blobs)
  return state
end

function forward(sys::System{CPU}, state::SoftmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input  = inputs[i]
    output = copy(state.blobs[i].data)

    # substract max before exp to avoid numerical issue
    data .-= max(output,2)
    output = exp(output)
    output ./= sum(output,2)
    blobs[i].data[:] = map(j -> indmax(output[j,:]), 1:size(output,1))
  end
end

# There is no backward procedure implemented. If you need to backpropagate
# from softmax classification, use SoftmaxLossLayer instead for better
# numerical stability.
