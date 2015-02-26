using HDF5

@defstruct AsyncHDF5DataLayer Layer (
  name :: String = "hdf5-data",
  (source :: String = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0),
  (tops :: Vector{Symbol} = Symbol[:data,:label], length(tops) > 0),
  shuffle :: Bool = false,
  transformers :: Vector = [],
)
@characterize_layer(AsyncHDF5DataLayer,
  is_source => true
)


type AsyncHDF5DataLayerState <: LayerState
  layer :: AsyncHDF5DataLayer
  blobs :: Vector{Blob}
  epoch :: Int
  trans :: Vector{Vector{DataTransformerState}}

  sources        :: Vector{String}
  shuffle_src    :: Vector{Int}

  io_task        :: Task
  stop_task      :: Bool

  AsyncHDF5DataLayerState(backend::Backend, layer::AsyncHDF5DataLayer) = begin
    state = new(layer)

    sources = open(layer.source, "r") do s
      map(strip, filter(l -> !isspace(l), readlines(s)))
    end
    @assert(length(sources) > 0)
    state.sources = sources
    state.epoch = 0

    if layer.shuffle
      state.shuffle_src = randperm(length(sources))
    else
      state.shuffle_src = collect(1:length(sources))
    end

    # empty array, will be constructed in setup
    state.blobs = Array(Blob, length(layer.tops))
    state.trans = Array(Vector{DataTransformerState}, length(layer.tops))

    return state
  end
end

function setup(backend::Backend, layer::AsyncHDF5DataLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  state = AsyncHDF5DataLayerState(backend, layer)

  h5_file = h5open(state.sources[state.shuffle_src[1]], "r")
  dsets = [h5_file[string(x)] for x in layer.tops]
  shuffle_idx = randperm(size(dsets[1],4))

  # setup blob shapes and data transformers
  transformers = convert(Vector{(Symbol, DataTransformerType)}, state.layer.transformers)
  for i = 1:length(state.blobs)
    dims = size(dsets[i])

    dims = tuple(dims[1:end-1]..., state.layer.batch_size)

    dset = dsets[i]
    state.blobs[i] = make_blob(backend, eltype(dset), dims)

    state.trans[i] = [setup(backend, convert(DataTransformerType, t), state.blobs[i])
        for (k,t) in filter(kt -> kt[1] == state.layer.tops[i], transformers)]
  end

  # start data reading task
  state.stop_task = false


  function io_task_impl()
    curr_index = 1
    curr_source = 1

    data_blocks = Array[Array(eltype(x), size(x)) for x in state.blobs]

    @info("AsyncHDF5DataLayer: IO Task starting...")
    while !state.stop_task
      # read and produce data
      n_done = 0
      while n_done < state.layer.batch_size
        n_remain = size(dsets[1])[end] - curr_index + 1
        if n_remain == 0
          # open next HDF5 file
          close(h5_file)
          curr_source = curr_source % length(state.sources) + 1
          h5_file = h5open(state.sources[state.shuffle_src[curr_source]], "r")
          dsets = [h5_file[string(x)] for x in state.layer.tops]

          if state.layer.shuffle
            shuffle_idx = randperm(size(dsets[1])[end])
          end

          curr_index = 1
          n_remain = size(dsets[1])[end]
        end

        n1 = min(state.layer.batch_size-n_done, n_remain)
        if n1 > 0
          for i = 1:length(state.blobs)
            idx = map(x -> 1:x, size(state.blobs[i])[1:end-1])
            dset = dsets[i]
            if state.layer.shuffle
              for kk = 1:n1
                data_blocks[i][idx...,n_done+kk] = dset[idx...,shuffle_idx[curr_index+kk-1]]
              end
            else
              data_blocks[i][idx...,n_done+1:n_done+n1] = dset[idx..., curr_index:curr_index+n1-1]
            end
          end
        end
        curr_index += n1
        n_done += n1

        # update epoch
        if curr_index > size(dsets[1])[end] && curr_source == length(state.sources)
          state.epoch += 1
        end
      end
      produce(data_blocks)
    end
    @info("AsyncHDF5DataLayer: IO Task reaching the end...")
    close(h5_file)
    produce(nothing)
  end

  # start the IO task
  state.io_task = Task(io_task_impl)

  return state
end
function shutdown(backend::Backend, state::AsyncHDF5DataLayerState)
  @info("AsyncHDF5DataLayer: Stopping IO task...")
  state.stop_task = true
  while consume(state.io_task) != nothing
    # ignore everything
  end

  map(destroy, state.blobs)
  map(ts -> map(t -> shutdown(backend, t), ts), state.trans)
end

function forward(backend::Backend, state::AsyncHDF5DataLayerState, inputs::Vector{Blob})
  data_blocks = consume(state.io_task)

  for i = 1:length(state.blobs)
    copy!(state.blobs[i], data_blocks[i])
    for j = 1:length(state.trans[i])
      forward(backend, state.trans[i][j], state.blobs[i])
    end
  end
end

function backward(backend::Backend, state::AsyncHDF5DataLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

