@defstruct ConcatLayer Layer (
  name :: String = "concat",
  (dim :: Int = 3, dim >= 1),
  (bottoms :: Vector{Symbol} = [], length(bottoms) >= 2),
  (tops :: Vector{Symbol} = [], length(tops) == 1)
)
@characterize_layer(ConcatLayer,
  can_do_bp  => true
)

type ConcatLayerState <: LayerState
  layer      :: ConcatLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::ConcatLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  # make sure all input blobs has the same tensor-dim
  for i = 2:length(inputs)
    @assert ndims(inputs[i]) == ndims(inputs[1])
  end
  tensor_dim = ndims(inputs[1])
  @assert layer.dim <= tensor_dim

  dims = map(size, inputs)
  dim_total = 0
  for i = 1:tensor_dim
    if i == layer.dim
      dim_total = sum(map(x -> x[i], dims))
    else
      # make sure the non-concating dims are equal
      for j = 2:length(dims)
        @assert dims[j][i] == dims[1][i]
      end
    end
  end

  my_dim = [dims[1]...]
  my_dim[layer.dim] = dim_total
  my_dim = tuple(my_dim...)
  data_type = eltype(inputs[1])

  blobs = Blob[make_blob(backend, data_type, my_dim)]
  blobs_diff = map(diffs) do blob
    if isa(blob, NullBlob)
      NullBlob()
    else
      make_blob(backend, data_type, my_dim)
    end
  end

  return ConcatLayerState(layer, blobs, blobs_diff)
end

function shutdown(backend::Backend, state::ConcatLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
end

function forward(backend::CPUBackend, state::ConcatLayerState, inputs::Vector{Blob})
  output = state.blobs[1]
  idx = map(x -> 1:x, [size(output)...])
  idx_start = 1
  for i = 1:length(inputs)
    idx_end = idx_start + size(inputs[i], state.layer.dim) - 1
    idx[state.layer.dim] = idx_start:idx_end
    setindex!(output.data, inputs[i].data, idx...)
    idx_start = idx_end + 1
  end
end

function backward(backend::CPUBackend, state::ConcatLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  idx = map(x -> 1:x, [size(state.blobs_diff[1])...])
  idx_start = 1
  for i = 1:length(diffs)
    idx_end = idx_start + size(inputs[i], state.layer.dim) - 1
    idx[state.layer.dim] = idx_start:idx_end
    if !isa(diffs[i], NullBlob)
      diffs[i].data[:] = getindex(state.blobs_diff[1].data, idx...)
    end
    idx_start = idx_end + 1
  end
end
