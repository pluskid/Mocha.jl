@defstruct ArgmaxLayer UtilLayer (
  name :: String = "argmax",
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)

type ArgmaxLayerState <: LayerState
  layer :: ArgmaxLayer
  blobs :: Vector{Blob}
end

function setup(sys::System, layer::ArgmaxLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs = map(inputs) do input
    width, height, channels, num = size(input)
    data_type = eltype(input)

    blob = make_blob(sys.backend, data_type, width, height, 1, num)
    blob
  end

  return ArgmaxLayerState(layer, blobs)
end

function forward(sys::System{CPUBackend}, state::ArgmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data
    width, height, channels, num = size(input)
    for n = 1:num
      for w = 1:width
        for h = 1:height
          maxc = 1; maxval = input[w,h,maxc,n]
          for c = 2:channels
            @inbounds val = input[w,h,c,n]
            if val > maxval
              maxval = val
              maxc = c
            end
          end
          @inbounds output[w,h,1,n] = maxc-1
        end
      end
    end
  end
end

function backward(sys::System, state::ArgmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # no backward for argmax layer
end

function shutdown(sys::System, state::ArgmaxLayerState)
end
