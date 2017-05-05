@defstruct ArgmaxLayer Layer (
  name :: AbstractString = "argmax",
  (dim :: Int = -2, dim != 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)

type ArgmaxLayerState <: LayerState
  layer :: ArgmaxLayer
  blobs :: Vector{Blob}

  dims  :: Vector{Int}
end

function setup(backend::Backend, layer::ArgmaxLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  dims = Array{Int}(length(inputs))
  blobs = Array{Blob}(length(inputs))
  for i = 1:length(inputs)
    total_dim = ndims(inputs[i])
    dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
    @assert 1 <= dim <= total_dim
    @assert dim != total_dim
    dims[i] = dim
    shape = [size(inputs[i])...]
    shape[dim] = 1
    blobs[i] = make_blob(backend, eltype(inputs[i]), shape...)
  end

  return ArgmaxLayerState(layer, blobs, dims)
end

function forward(backend::CPUBackend, state::ArgmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data
    pre_dim, mid_dim, post_dim = split_dims(input, state.dims[i])
    for x = 0:pre_dim-1
      for z = 0:post_dim-1
        idx = Int[x + pre_dim*(y + mid_dim*z) for y=0:mid_dim-1] + 1
        maxc = 1
        @inbounds maxval = input[idx[1]]
        for y = 2:length(idx)
          @inbounds val = input[idx[y]]
          if val > maxval
            maxval = val
            maxc = y
          end
        end
        @inbounds output[x + pre_dim*z + 1] = maxc-1
      end
    end
  end
end

function backward(backend::Backend, state::ArgmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # no backward for argmax layer
end

function shutdown(backend::Backend, state::ArgmaxLayerState)
end
