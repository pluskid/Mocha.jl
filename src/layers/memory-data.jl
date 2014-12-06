@defstruct MemoryDataLayer Layer (
  name :: String = "memory-data",
  (tops :: Vector{Symbol} = Symbol[:data,:label], length(tops) > 0),
  (batch_size :: Int = 0, batch_size > 0),
  (data :: Vector{Array} = Array[], length(data) == length(tops)),
  transformers :: Vector = [],
)
@characterize_layer(MemoryDataLayer,
  is_source => true
)

type MemoryDataLayerState <: LayerState
  layer :: MemoryDataLayer
  blobs :: Vector{Blob}
  epoch :: Int
  trans :: Vector{Vector{DataTransformerState}}

  curr_idx :: Int

  MemoryDataLayerState(backend::Backend, layer::MemoryDataLayer) = begin
    blobs = Array(Blob, length(layer.tops))
    trans = Array(Vector{DataTransformerState}, length(layer.tops))
    transformers = convert(Vector{(Symbol, DataTransformerType)}, layer.transformers)
    for i = 1:length(blobs)
      dims = tuple(size(layer.data[i])[1:3]..., layer.batch_size)
      idxs = map(x -> 1:x, dims)

      blobs[i] = make_blob(backend, eltype(layer.data[i]), dims...)
      trans[i] = [setup(backend, convert(DataTransformerType, t), blobs[i])
          for (k,t) in filter(kt -> kt[1] == layer.tops[i], transformers)]
    end

    new(layer, blobs, 0, trans, 1)
  end
end

function setup(backend::Backend, layer::MemoryDataLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  for i = 1:length(layer.data)
    dims = size(layer.data[i])
    if length(dims) > 4
      error("Tensor dimension in data $(layer.tops[i]): $(length(dims)) > 4")
    elseif length(dims) < 4
      dims = tuple(ones(Int, 4-length(dims))..., dims...)
      layer.data[i] = reshape(layer.data[i], dims)
    end
  end
  state = MemoryDataLayerState(backend, layer)
  return state
end
function shutdown(backend::Backend, state::MemoryDataLayerState)
  map(destroy, state.blobs)
  map(ts -> map(t -> shutdown(backend, t), ts), state.trans)
end

function forward(backend::Backend, state::MemoryDataLayerState, inputs::Vector{Blob})
  n_done = 0
  while n_done < state.layer.batch_size
    n_remain = size(state.layer.data[1], 4) - state.curr_idx + 1
    if n_remain == 0
      state.curr_idx = 1
      n_remain = size(state.layer.data[1], 4)
    end

    n1 = min(state.layer.batch_size - n_done, n_remain)
    for i = 1:length(state.blobs)
      dset = state.layer.data[i]
      idx = map(x -> 1:x, size(state.blobs[i])[1:3])
      the_data = dset[idx..., state.curr_idx:state.curr_idx+n1-1]
      set_blob_data(the_data, state.blobs[i], n_done+1)
    end
    state.curr_idx += n1
    n_done += n1

    if state.curr_idx > size(state.layer.data[1], 4)
      state.epoch += 1
    end
  end

  for i = 1:length(state.blobs)
    for j = 1:length(state.trans[i])
      forward(backend, state.trans[i][j], state.blobs[i])
    end
  end
end

function backward(backend::Backend, state::MemoryDataLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

