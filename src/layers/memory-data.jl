@defstruct MemoryDataLayer DataLayer (
  (batch_size :: Int = 0, batch_size > 0),
  (tops :: Vector{String} = String["data","label"], length(tops) > 0),
  (data :: Vector{Array} = Array[], length(data) == length(tops))
)

type MemoryDataLayerState <: LayerState
  layer :: MemoryDataLayer
  blobs :: Vector{Blob}

  curr_idx :: Int

  MemoryDataLayerState(sys::System, layer::MemoryDataLayer) = begin
    blobs = Array(CPUBlob, length(layer.tops))
    for i = 1:length(blobs)
      dims = tuple(size(layer.data[i])[1:3]..., layer.batch_size)
      idxs = map(x -> 1:x, dims)

      if isa(sys.backend, CPUBackend)
        blobs[i] = CPUBlob(Array(eltype(layer.data[i]), dims))
      elseif isa(sys.backend, CuDNNBackend)
        blobs[i] = cudnn_make_tensor_blob(eltype(layer.data[i]), dims...)
      else
        error("Backend $(sys.backend) not supported")
      end
    end

    new(layer, blobs, 1)
  end
end

function setup(sys::System, layer::MemoryDataLayer, inputs::Vector{Blob})
  @assert length(inputs) == 0
  state = MemoryDataLayerState(sys, layer)
  return state
end

function forward(sys::System, state::MemoryDataLayerState, inputs::Vector{Blob})
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
      idx = map(x -> 1:x, size(state.blobs[i].data)[1:3])
      the_data = dset[idx..., state.curr_idx:state.curr_idx+n1-1]
      set_blob_data(the_data, state.blobs[i], n_done+1)
    end
    state.curr_idx += n1
    n_done += n1
  end
end

