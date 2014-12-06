@defstruct ChannelPoolingLayer Layer (
  name :: String = "channel-pooling",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: Int = 1, kernel > 0),
  (stride :: Int = 1, stride > 0),
  (pad :: NTuple{2, Int} = (0,0), all([pad...] .>= 0)),
  pooling :: PoolingFunction = Pooling.Max(),
)
@characterize_layer(ChannelPoolingLayer,
  can_do_bp => true,
)

type ChannelPoolingLayerState <: LayerState
  layer      :: ChannelPoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  etc        :: Any
end

function setup_etc(backend::CPUBackend, layer::ChannelPoolingLayer, inputs, pooled_chann)
  if isa(layer.pooling, Pooling.Max)
    masks = Array(Array, length(inputs))
    for i = 1:length(inputs)
      masks[i] = Array(Csize_t, get_width(inputs[i]), get_height(inputs[i]),
          pooled_chann, get_num(inputs[i]))
    end
    etc = masks
  elseif isa(layer.pooling, Pooling.Mean)
    integrals = Array(Array, length(inputs))
    for i = 1:length(inputs)
      integrals[i] = Array(eltype(inputs[i]), get_width(inputs[i]), get_height(inputs[i]),
          get_chann(inputs[i]))
    end
    etc = integrals
  else
    etc = nothing
  end
  return etc
end

function setup(backend::Backend, layer::ChannelPoolingLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  width, height, channels, num = size(inputs[1])
  pooled_chann = int(ceil(float(channels + layer.pad[1]+layer.pad[2] - layer.kernel) / layer.stride)) + 1

  # make sure the last pooling is not purely pooling padded area
  if ((pooled_chann-1)*layer.stride >= channels + layer.pad[1])
    pooled_chann -= 1
  end

  data_type = eltype(inputs[1])
  blobs = Array(Blob, length(inputs))
  blobs_diff = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    blobs[i] = make_blob(backend, data_type, (width,height,pooled_chann,num))
    if isa(diffs[i], NullBlob)
      blobs_diff[i] = NullBlob()
    else
      blobs_diff[i] = make_blob(backend, data_type, (width,height,pooled_chann,num))
    end
  end

  etc = setup_etc(backend, layer, inputs, pooled_chann)
  state = ChannelPoolingLayerState(layer, blobs, blobs_diff, etc)
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

    if isa(pool, Pooling.Max)
      max_channel_pooling_forward(input, output, state.etc[i], state.layer)
    elseif isa(pool, Pooling.Mean)
      mean_channel_pooling_forward(input, output, state.etc[i], state.layer)
    else
      error("Pooling for $pool not implemented yet")
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
      if isa(pool, Pooling.Max)
        max_channel_pooling_backward(diff.data, state.blobs_diff[i].data, state.etc[i], state.layer)
      elseif isa(pool, Pooling.Mean)
        mean_channel_pooling_backward(diff.data, state.blobs_diff[i].data, state.layer)
      else
        error("Pooling for $pool not implemented yet")
      end
    else
      continue # nothing to do if not propagating back
    end
  end
end

