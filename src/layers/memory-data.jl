@defstruct MemoryDataLayer DataLayer (
  (batch_size :: Int = 0, batch_size > 0), 
  (tops :: Vector{String} = String["data","label"], length(tops) > 0),
  (data :: Vector{Array} = Array[], length(data) == length(tops))
)

type MemoryDataLayerState <: LayerState
  layer :: MemoryDataLayer
  blobs :: Vector{Blob}

  curr_idx :: Int

  MemoryDataLayerState(layer::MemoryDataLayer) = begin
    blobs = Array(CPUBlob, length(layer.tops))
    for i = 1:length(blobs)
      dims = tuple(layer.batch_size, size(layer.data[i])[2:end]...)
      idxs = map(x -> 1:x, dims)

      blobs[i] = CPUBlob(layer.tops[i], layer.data[i][idxs...])
    end

    new(layer, blobs, 1)
  end
end

function setup(layer::MemoryDataLayer, inputs::Vector{Blob})
  @assert length(inputs) == 0
  state = MemoryDataLayerState(layer)
  return state
end

function forward(state::HDF5DataLayerState)
  n_done = 0
  while n_done < state.layer.batch_size
    n_remain = size(state.layer.data[1], 1) - state.curr_idx + 1
    if n_remain == 0
      state.curr_idx = 1
      n_remain = size(state.layer.data[1],1)
    end

    n1 = min(state.layer.batch_size - n_done, n_remain)
    for i = 1:length(state.blobs)
      idx = map(x -> 1:x, size(state.blobs[i].data)[2:end])
      dset = state.layer.data[i]
      state.blobs[i].data[n_done+1:n_done+n1,idx...] = dset[state.curr_idx:state.curr_idx+n1-1,idx...]
    end
    state.curr_idx += n1
    n_done += n1
  end
end
