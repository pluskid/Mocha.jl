using HDF5

@defstruct HDF5DataLayer DataLayer (
  (source :: String = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0), 
  tops :: Vector{String} = String["data","label"]
)

type HDF5DataLayerState <: LayerState
  layer :: HDF5DataLayer
  blobs :: Vector{Blob}

  sources        :: Vector{String}
  curr_source    :: Int
  curr_hdf5_file :: HDF5File
  curr_index     :: Int

  HDF5DataLayerState(layer) = begin
    state = new(layer)

    sources = open(layer.source, "r") do s
      map(strip, filter(l -> !isspace(l), readlines(s)))
    end
    @assert(length(sources) > 0)
    state.sources = sources

    state.curr_source = 1
    state.curr_hdf5_file = h5open(sources[1], "r")

    state.blobs = Array(Blob, length(layer.tops))
    for i = 1:length(state.blobs)
      dims = size(state.curr_hdf5_file[layer.tops[i]])
      if layer.batch_size > 0
        dims = tuple(layer.batch_size, dims[2:end]...)
      end

      idx = [1:x for x in dims]
      dset = state.curr_hdf5_file[layer.tops[i]]
      state.blobs[i] = CPUBlob(layer.tops[i], dset[idx...])
    end
    state.curr_index = 1

    return state
  end
end

function setup(layer::HDF5DataLayer, inputs::Vector{Blob})
  @assert length(inputs) == 0
  state = HDF5DataLayerState(layer)
  return state
end

function forward(state::HDF5DataLayerState)
  idx = map(x -> 1:x, size(state.blobs[1].data)[2:end])
  n_done = 0
  while n_done < state.layer.batch_size
    n_remain = size(state.curr_hdf5_file[state.layer.tops[1]])[1] - state.curr_index + 1
    if n_remain == 0
      close(state.curr_hdf5_file)
      state.curr_source = state.curr_source % length(state.sources) + 1
      state.curr_hdf5_file = h5open(state.sources[state.curr_source], "r")
      state.curr_index = 1
      n_remain = size(state.curr_hdf5_file[state.layer.tops[1]])[1]
    end

    n1 = min(state.layer.batch_size-n_done, n_remain)

    if n1 > 0
      for i = 1:length(state.blobs)
        dset = state.curr_hdf5_file[state.layer.tops[i]]
        state.blobs[i].data[n_done+1:n_done+n1, idx...] = dset[state.curr_index:state.curr_index+n1-1, idx...]
      end
    end
    state.curr_index += n1
    n_done += n1
  end
end

