using HDF5

@defstruct HDF5DataLayer DataLayer (
  (source :: String = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0),
  (tops :: Vector{String} = String["data","label"], length(tops) > 0)
)

type HDF5DataLayerState <: LayerState
  layer :: HDF5DataLayer
  blobs :: Vector{Blob}

  sources        :: Vector{String}
  curr_source    :: Int
  curr_hdf5_file :: HDF5File
  curr_index     :: Int

  HDF5DataLayerState(sys::System, layer::HDF5DataLayer) = begin
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
      dims = tuple(dims[1:3]..., layer.batch_size)

      dset = state.curr_hdf5_file[layer.tops[i]]
      if isa(sys.backend, CPUBackend)
        state.blobs[i] = CPUBlob(eltype(dset), dims)
      elseif isa(sys.backend, CuDNNBackend)
        state.blobs[i] = cudnn_make_tensor_blob(eltype(dset), dims...)
      else
        error("Backend $(sys.backend) not supported")
      end
    end
    state.curr_index = 1

    return state
  end
end

function setup(sys::System, layer::HDF5DataLayer, inputs::Vector{Blob})
  @assert length(inputs) == 0
  state = HDF5DataLayerState(sys, layer)
  return state
end

function forward(sys::System, state::HDF5DataLayerState, inputs::Vector{Blob})
  n_done = 0
  while n_done < state.layer.batch_size
    n_remain = size(state.curr_hdf5_file[state.layer.tops[1]])[4] - state.curr_index + 1
    if n_remain == 0
      close(state.curr_hdf5_file)
      state.curr_source = state.curr_source % length(state.sources) + 1
      state.curr_hdf5_file = h5open(state.sources[state.curr_source], "r")
      state.curr_index = 1
      n_remain = size(state.curr_hdf5_file[state.layer.tops[1]])[4]
    end

    n1 = min(state.layer.batch_size-n_done, n_remain)

    if n1 > 0
      for i = 1:length(state.blobs)
        idx = map(x -> 1:x, size(state.blobs[i].data)[1:3])
        dset = state.curr_hdf5_file[state.layer.tops[i]]
        the_data = dset[idx..., state.curr_index:state.curr_index+n1-1]
        set_blob_data(the_data, state.blobs[i], n_done+1)
      end
    end
    state.curr_index += n1
    n_done += n1
  end
end

function set_blob_data(data::Array, blob::CPUBlob, blob_idx::Int)
  n_fea = prod(size(blob)[1:3])
  idx_start = (blob_idx-1)*n_fea
  blob.data[idx_start+1:idx_start+length(data)] = data
end
function set_blob_data{T}(data::Array{T}, blob::CuTensorBlob{T}, blob_idx::Int)
  ptr = convert(Ptr{Void}, blob.ptr.p) + sizeof(T) * (blob_idx-1) # note 0-based indexing in CUDA Vector
  CuBLAS.set_vector(length(data), sizeof(T), convert(Ptr{Void},data), 1, ptr, 1)
end
