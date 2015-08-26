using HDF5

@defstruct HDF5DataLayer Layer (
  name :: String = "hdf5-data",
  (source :: String = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0),
  (tops :: Vector{Symbol} = Symbol[:data,:label], length(tops) > 0),
  shuffle :: Bool = false,
  transformers :: Vector = [],
)
@characterize_layer(HDF5DataLayer,
  is_source => true
)


type HDF5DataLayerState <: LayerState
  layer :: HDF5DataLayer
  blobs :: Vector{Blob}
  epoch :: Int
  trans :: Vector{Vector{DataTransformerState}}

  sources        :: Vector{String}
  dsets          :: Vector{Any}
  curr_source    :: Int
  curr_hdf5_file :: HDF5File
  curr_index     :: Int
  shuffle_idx    :: Vector{Int}
  shuffle_src    :: Vector{Int}

  HDF5DataLayerState(backend::Backend, layer::HDF5DataLayer) = begin
    state = new(layer)

    sources = open(layer.source, "r") do s
      map(strip, filter(l -> !isspace(l), readlines(s)))
    end
    @assert(length(sources) > 0)
    state.sources = sources

    if layer.shuffle
      state.shuffle_src = randperm(length(state.sources))
    else
      state.shuffle_src = collect(1:length(state.sources))
    end

    state.epoch = 0
    state.curr_source = 1
    state.curr_hdf5_file = h5open(sources[state.shuffle_src[1]], "r")

    state.dsets = [state.curr_hdf5_file[string(x)] for x in layer.tops]
    if layer.shuffle
      # we only use mmap when random shuffle is required, as mmap is slower
      # than sequential batch read. see hdf5-read benchmark.
      state.dsets = map(readmmap, state.dsets)
      state.shuffle_idx = randperm(size(state.dsets[1])[end])
    else
      state.shuffle_idx = Int[]
    end

    state.blobs = Array(Blob, length(layer.tops))
    state.trans = Array(Vector{DataTransformerState}, length(layer.tops))
    transformers = convert(Vector{@compat(Tuple{Symbol, DataTransformerType})}, layer.transformers)
    for i = 1:length(state.blobs)
      dims = size(state.dsets[i])

      dims = tuple(dims[1:end-1]..., layer.batch_size)

      dset = state.dsets[i]
      state.blobs[i] = make_blob(backend, eltype(dset), dims)

      state.trans[i] = [setup(backend, convert(DataTransformerType, t), state.blobs[i])
          for (k,t) in filter(kt -> kt[1] == layer.tops[i], transformers)]
    end
    state.curr_index = 1

    return state
  end
end

function setup(backend::Backend, layer::HDF5DataLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  state = HDF5DataLayerState(backend, layer)
  return state
end
function shutdown(backend::Backend, state::HDF5DataLayerState)
  map(destroy, state.blobs)
  map(ts -> map(t -> shutdown(backend, t), ts), state.trans)
  close(state.curr_hdf5_file)
end

function forward(backend::Backend, state::HDF5DataLayerState, inputs::Vector{Blob})
  n_done = 0
  while n_done < state.layer.batch_size
    n_remain = size(state.dsets[1])[end] - state.curr_index + 1
    if n_remain == 0
      close(state.curr_hdf5_file)
      state.curr_source = state.curr_source % length(state.sources) + 1
      state.curr_hdf5_file = h5open(state.sources[state.shuffle_src[state.curr_source]], "r")
      state.dsets = [state.curr_hdf5_file[string(x)] for x in state.layer.tops]

      if state.layer.shuffle
        state.dsets = map(readmmap, state.dsets)
        state.shuffle_idx = randperm(size(state.dsets[1])[end])
      end

      state.curr_index = 1
      n_remain = size(state.dsets[1])[end]
    end

    n1 = min(state.layer.batch_size-n_done, n_remain)
    if n1 > 0
      for i = 1:length(state.blobs)
        idx = map(x -> 1:x, size(state.blobs[i])[1:end-1])
        dset = state.dsets[i]
        if state.layer.shuffle
          the_data = dset[idx..., state.shuffle_idx[state.curr_index:state.curr_index+n1-1]]
        else
          the_data = dset[idx..., state.curr_index:state.curr_index+n1-1]
        end
        set_blob_data(the_data, state.blobs[i], n_done+1)
      end
    end
    state.curr_index += n1
    n_done += n1

    # update epoch
    if state.curr_index > size(state.dsets[1])[end] &&
        state.curr_source == length(state.sources)
      state.epoch += 1
    end
  end

  for i = 1:length(state.blobs)
    for j = 1:length(state.trans[i])
      forward(backend, state.trans[i][j], state.blobs[i])
    end
  end
end

function set_blob_data(data::Array, blob::CPUBlob, blob_idx::Int)
  n_fea = get_fea_size(blob)
  idx_start = (blob_idx-1)*n_fea
  blob.data[idx_start+1:idx_start+length(data)] = data
end

function backward(backend::Backend, state::HDF5DataLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end
