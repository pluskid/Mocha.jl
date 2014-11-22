############################################################
# Softmax Layer
############################################################
@defstruct SoftmaxLayer CompLayer (
  name :: String = "softmax",
  tops :: Vector{Symbol} = Symbol[],
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops))
)

type SoftmaxLayerState <: LayerState
  layer      :: SoftmaxLayer
  blobs      :: Vector{Blob}

  etc        :: Any
end

function setup_etc(sys::System{CPUBackend}, layer::SoftmaxLayer, data_type, inputs)
  nothing
end

function setup(sys::System, layer::SoftmaxLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type  = eltype(inputs[1])
  blobs      = Blob[make_blob(sys.backend, data_type, size(input)) for input in inputs]
  etc        = setup_etc(sys, layer, data_type, inputs)

  state = SoftmaxLayerState(layer, blobs, etc)
  return state
end
function shutdown(sys::System{CPUBackend}, state::SoftmaxLayerState)
  map(destroy, state.blobs)
end

function forward(sys::System{CPUBackend}, state::SoftmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input  = inputs[i].data
    output = state.blobs[i].data

    width, height, channels, num = size(input)

    for w = 1:width
      for h = 1:height
        for n = 1:num
          maxval = -Inf
          for c = 1:channels
            @inbounds maxval = max(maxval, input[w,h,c,n])
          end
          for c = 1:channels
            @inbounds output[w,h,c,n] = exp(input[w,h,c,n]-maxval)
          end
          the_sum = 0.0
          for c = 1:channels
            @inbounds the_sum += output[w,h,c,n]
          end
          for c = 1:channels
            @inbounds output[w,h,c,n] /= the_sum
          end
        end
      end
    end
  end
end

function backward(sys::System, state::SoftmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

