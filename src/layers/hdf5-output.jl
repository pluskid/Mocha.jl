using HDF5

@defstruct HDF5OutputLayer Layer (
  name :: AbstractString = "hdf5-output",
  (bottoms :: Vector{Symbol} = [], length(bottoms) > 0),
  (datasets :: Vector{Symbol} = Symbol[], length(datasets) == 0 || length(datasets) == length(bottoms)),
  (filename :: AbstractString = "", !isempty(filename)),
  force_overwrite :: Bool = false,
)
@characterize_layer(HDF5OutputLayer,
  is_sink => true
)

type HDF5OutputLayerState <: LayerState
  layer  :: HDF5OutputLayer
  file   :: HDF5File
  buffer :: Vector{Array}
  dsets  :: Vector{Any}
  index  :: Int
end

function setup(backend::Backend, layer::HDF5OutputLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  if isfile(layer.filename)
    if !layer.force_overwrite
      error("HDF5OutputLayer: output file '$(layer.filename)' already exists")
    else
      m_warn("HDF5OutputLayer: output file '$(layer.filename)' already exists, overwriting")
    end
  end
  if length(layer.datasets) == 0
    datasets = layer.bottoms
  else
    datasets = layer.datasets
  end

  file = h5open(layer.filename, "w")
  buffer = Array{Array}(length(inputs))
  dsets  = Array{Any}(length(inputs))
  for i = 1:length(inputs)
    data_type = eltype(inputs[i])
    dims = size(inputs[i])
    dsets[i] = d_create(file, string(datasets[i]), datatype(data_type),
        dataspace(dims, max_dims=tuple(dims[1:end-1]..., -1)), "chunk", dims)
    buffer[i] = Array{data_type}(dims)
  end

  return HDF5OutputLayerState(layer, file, buffer, dsets, 1)
end

function forward(backend::Backend, state::HDF5OutputLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    copy!(state.buffer[i], inputs[i])
    dims = size(state.buffer[i])
    batch_size = dims[end]

    # extend the HDF5 dataset
    set_dims!(state.dsets[i], tuple(dims[1:end-1]..., state.index*batch_size))

    # write data
    idx = map(x -> 1:x, dims[1:end-1])
    state.dsets[i][idx...,(state.index-1)*batch_size+1:state.index*batch_size] = state.buffer[i]
  end
  state.index += 1
end

function backward(backend::Backend, state::HDF5OutputLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

function shutdown(backend::Backend, state::HDF5OutputLayerState)
  close(state.file)
end
