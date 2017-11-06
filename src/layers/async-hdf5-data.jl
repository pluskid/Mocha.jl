using HDF5

@defstruct AsyncHDF5DataLayer Layer (
  name :: AbstractString = "hdf5-data",
  (source :: AbstractString = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0),
  (chunk_size :: Int = 2^20, chunk_size > 0),
  (tops :: Vector{Symbol} = Symbol[:data,:label], length(tops) > 0),
  shuffle :: Bool = false,
  transformers :: Vector = [],
)
@characterize_layer(AsyncHDF5DataLayer,
  is_source => true
)

const AsyncCommsType = @static VERSION < v"0.6-" ? Task : Channel{Any}

type AsyncHDF5DataLayerState <: LayerState
  layer :: AsyncHDF5DataLayer
  blobs :: Vector{Blob}
  epoch :: Int
  trans :: Vector{Vector{DataTransformerState}}

  sources        :: Vector{AbstractString}

  io_channel     :: AsyncCommsType
  stop_task      :: Bool

  AsyncHDF5DataLayerState(backend::Backend, layer::AsyncHDF5DataLayer) = begin
    state = new(layer)

    sources = open(layer.source, "r") do s
      map(strip, filter(l -> !all(isspace, l), readlines(s)))
    end
    @assert(length(sources) > 0)
    state.sources = sources
    state.epoch = 0

    # empty array, will be constructed in setup
    state.blobs = Array{Blob}(length(layer.tops))
    state.trans = Array{Vector{DataTransformerState}}(length(layer.tops))

    return state
  end
end

function setup(backend::Backend, layer::AsyncHDF5DataLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  state = AsyncHDF5DataLayerState(backend, layer)

  h5_file = h5open(state.sources[1], "r")
  dsets = [h5_file[string(x)] for x in layer.tops]

  # setup blob shapes and data transformers
  transformers = convert(Vector{@compat(Tuple{Symbol, DataTransformerType})}, state.layer.transformers)
  for i = 1:length(state.blobs)
    dims = size(dsets[i])

    dims = tuple(dims[1:end-1]..., state.layer.batch_size)

    dset = dsets[i]
    state.blobs[i] = make_blob(backend, eltype(dset), dims)

    state.trans[i] = [setup(backend, convert(DataTransformerType, t), state.blobs[i])
        for (k,t) in filter(kt -> kt[1] == state.layer.tops[i], transformers)]
  end

  state.stop_task = false
  close(h5_file)


  function io_task_impl(channel)
    # data blocks to produce
    data_blocks = Array[Array{eltype(x)}(size(x)) for x in state.blobs]
    n_done = 0

    while true
      if layer.shuffle
        shuffle_src = randperm(length(state.sources))
      else
        shuffle_src = collect(1:length(state.sources))
      end

      for i_file = 1:length(shuffle_src)
        h5_file = h5open(state.sources[shuffle_src[i_file]], "r")
        dsets = [h5_file[string(x)] for x in layer.tops]

        n_total = size(dsets[1])[end]
        chunk_idx = 1:layer.chunk_size:n_total
        if layer.shuffle
          shuffle_chunk = randperm(length(chunk_idx))
        else
          shuffle_chunk = collect(1:length(chunk_idx))
        end

        # load data in chunks
        for i_chunk = 1:length(chunk_idx)
          idx_start = chunk_idx[shuffle_chunk[i_chunk]]
          idx_end = min(idx_start+layer.chunk_size-1, n_total)

          data_chunks = map(1:length(dsets)) do i_dset
            idx = map(x -> 1:x, size(state.blobs[i_dset])[1:end-1])
            dsets[i_dset][idx..., idx_start:idx_end]
          end

          n_chunk = idx_end - idx_start+1
          if layer.shuffle
            # shuffle within chunk
            shuffle_idx = randperm(n_chunk)
          end

          curr_idx = 1
          while curr_idx <= n_chunk
            if n_done == layer.batch_size
              @static if VERSION < v"0.6-"
                produce(data_blocks)
              else
                put!(channel, data_blocks)
              end
              if state.stop_task
                m_info("AsyncHDF5DataLayer: IO Task reaching the end...")
                close(h5_file)
                @static if VERSION < v"0.6-"
                  produce(nothing)
                else
                  put!(channel, nothing)
                end
                return
              end

              n_done = 0
            end
            n_todo = layer.batch_size - n_done

            n_take = min(n_todo, n_chunk-curr_idx+1)
            for i_dset = 1:length(dsets)
              idx = map(x -> 1:x, size(state.blobs[i_dset])[1:end-1])
              idx_take = curr_idx:curr_idx+n_take-1
              if layer.shuffle
                idx_take = shuffle_idx[idx_take]
              end
              data_blocks[i_dset][idx...,n_done+1:n_done+n_take] = data_chunks[i_dset][idx..., idx_take]
            end
            curr_idx += n_take
            n_done += n_take
          end
        end
      end

      # update epoch
      state.epoch += 1
    end
  end

  # start the IO task
  @static if VERSION < v"0.6-"
    state.io_channel = Task(() -> io_task_impl(nothing))
  else
    state.io_channel = Channel(io_task_impl)
  end
  # state.io_task = Task(io_task_impl)

  return state
end
function shutdown(backend::Backend, state::AsyncHDF5DataLayerState)
  m_info("AsyncHDF5DataLayer: Stopping IO task...")
  state.stop_task = true
  while take!(state.io_channel) != nothing
    # ignore everything
  end

  map(destroy, state.blobs)
  map(ts -> map(t -> shutdown(backend, t), ts), state.trans)
end

function forward(backend::Backend, state::AsyncHDF5DataLayerState, inputs::Vector{Blob})
  data_blocks = take!(state.io_channel)

  for i = 1:length(state.blobs)
    copy!(state.blobs[i], data_blocks[i])
    for j = 1:length(state.trans[i])
      forward(backend, state.trans[i][j], state.blobs[i])
    end
  end
end

function backward(backend::Backend, state::AsyncHDF5DataLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

