@defstruct ChannelPoolingLayer Layer (
  name :: String = "channel-pooling",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: Int = 1, kernel > 0),
  (stride :: Int = 1, stride > 0),
  (pad :: NTuple{2, Int} = (0,0), all([pad...] .>= 0)),
  (channel_dim :: Int = -2, channel_dim != 0),
  pooling :: PoolingFunction = Pooling.Max(),
)
@characterize_layer(ChannelPoolingLayer,
  can_do_bp => true,
)

type ChannelPoolingLayerState <: LayerState
  layer      :: ChannelPoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  op_dims    :: Vector{Int}
  etc        :: Any
end

function setup_etc(backend::CPUBackend, layer::ChannelPoolingLayer, inputs, blobs)
  if isa(layer.pooling, Pooling.Max)
    masks = Array(Array, length(inputs))
    for i = 1:length(inputs)
      masks[i] = Array(Csize_t, size(blobs[i]))
    end
    etc = masks
  elseif isa(layer.pooling, Pooling.Mean)
    integrals = Array(Array, length(inputs))
    for i = 1:length(inputs)
      integrals[i] = Array(eltype(inputs[i]), size(inputs[i])[1:end-1])
    end
    etc = integrals
  else
    etc = nothing
  end
  return etc
end

function setup(backend::Backend, layer::ChannelPoolingLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  pooled_chann_all = Array(Int, length(inputs))
  blobs = Array(Blob, length(inputs))
  blobs_diff = Array(Blob, length(inputs))
  op_dims = Array(Int, length(inputs))

  for i = 1:length(inputs)
    dim_total = ndims(inputs[i])
    op_dim = layer.channel_dim < 0 ? layer.channel_dim + dim_total+1 : layer.channel_dim
    @assert 1 <= op_dim <= dim_total
    @assert op_dim != dim_total

    op_dims[i] = op_dim

    dims = [size(inputs[i])...]
    pool_dim = dims[op_dim]
    pooled_dim = round(Int64, ceil(float(pool_dim + layer.pad[1]+layer.pad[2] - layer.kernel) / layer.stride)) + 1

    # make sure the last pooling is not purely pooling padded area
    if ((pooled_dim-1)*layer.stride >= pool_dim + layer.pad[1])
      pooled_dim -= 1
    end
    pooled_chann_all[i] = pooled_dim

    output_dims = copy(dims)
    output_dims[op_dim] = pooled_dim
    output_dims = tuple(output_dims...)

    data_type = eltype(inputs[i])
    blobs[i] = make_blob(backend, data_type, output_dims)
    if isa(diffs[i], NullBlob)
      blobs_diff[i] = NullBlob()
    else
      blobs_diff[i] = make_blob(backend, data_type, output_dims)
    end
  end

  etc = setup_etc(backend, layer, inputs, blobs)
  state = ChannelPoolingLayerState(layer, blobs, blobs_diff, op_dims, etc)
end
function shutdown_etc(backend::CPUBackend, state::ChannelPoolingLayerState)
end
function shutdown(backend::Backend, state::ChannelPoolingLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  shutdown_etc(backend, state)
end

function forward(backend::CPUBackend, state::ChannelPoolingLayerState, inputs::Vector{Blob})
  forward(backend, state.layer.pooling, state, inputs)
end
function forward(backend::CPUBackend, pool::StdPoolingFunction,
    state::ChannelPoolingLayerState, inputs::Vector{Blob})

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data

    dims_in = split_dims(input, state.op_dims[i])
    dims_out = split_dims(output, state.op_dims[i])

    if isa(pool, Pooling.Max)
      max_channel_pooling_forward(reshape(input,dims_in), reshape(output,dims_out), reshape(state.etc[i],dims_out), state.layer)
    elseif isa(pool, Pooling.Mean)
      mean_channel_pooling_forward(reshape(input,dims_in), reshape(output,dims_out), state.etc[i], state.layer)
    end
  end
end

function backward(backend::CPUBackend, state::ChannelPoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(backend, state.layer.pooling, state, inputs, diffs)
end

function backward(backend::CPUBackend, pool::StdPoolingFunction, state::ChannelPoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(inputs)
    diff = diffs[i]

    if !isa(diff, NullBlob)
      dims_in = split_dims(inputs[i], state.op_dims[i])
      dims_out = split_dims(state.blobs[i], state.op_dims[i])

      if isa(pool, Pooling.Max)
        max_channel_pooling_backward(reshape(diff.data,dims_in), reshape(state.blobs_diff[i].data,dims_out),
            reshape(state.etc[i],dims_out), state.layer)
      elseif isa(pool, Pooling.Mean)
        mean_channel_pooling_backward(reshape(diff.data,dims_in), reshape(state.blobs_diff[i].data,dims_out), state.layer)
      end
    end
  end
end

