@defstruct Index2OnehotLayer Layer (
  name :: String = "index2onehot",
  (dim :: Int = -2, dim != 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
  (n_class :: Int = 0, n_class > 0)
)

type Index2OnehotLayerState <: LayerState
  layer :: Index2OnehotLayer
  blobs :: Vector{Blob}

  dims  :: Vector{Int}
end

function setup(backend::Backend, layer::Index2OnehotLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type  = eltype(inputs[1])
  dims = Array(Int, length(inputs))
  blobs = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    total_dim = ndims(inputs[i])
    dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
    @assert 1 <= dim <= total_dim
    @assert dim != total_dim # should not operate on the mini-batch dimension

    dims[i] = dim
    blob_size = collect(size(inputs[i]))
    @assert blob_size[dim] == 1
    blob_size[dim] = layer.n_class
    blobs[i] = make_blob(backend, data_type, tuple(blob_size...))
  end
  return Index2OnehotLayerState(layer, blobs, dims)
end

function shutdown(backend::Backend, state::Index2OnehotLayerState)
  map(destroy, state.blobs)
end

function forward(backend::CPUBackend, state::Index2OnehotLayerState, inputs::Vector{Blob})
  for ii = 1:length(inputs)
    erase!(state.blobs[ii])

    input = inputs[ii].data
    output = state.blobs[ii].data
    op_dim = state.dims[ii]

    dim_pre, dim_prob, dim_post = split_dims(input, op_dim)
    @assert dim_prob == 1
    for i = 0:dim_pre-1
      for j = 0:dim_post-1
        @inbounds label = convert(Int, input[i + dim_pre*j + 1])
        idx = i + dim_pre*(label + state.layer.n_class*j) + 1
        @inbounds output[idx] = 1
      end
    end
  end
end

function backward(backend::Backend, state::Index2OnehotLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end
