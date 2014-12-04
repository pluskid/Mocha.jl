@defstruct PoolingLayer CompLayer (
  name :: String = "pooling",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: NTuple{2, Int} = (1,1), all([kernel...] .> 0)),
  (stride :: NTuple{2, Int} = (1,1), all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), all([pad...] .>= 0)),
  pooling :: PoolingFunction = Pooling.Max(),
  neuron :: ActivationFunction = Neurons.Identity(),
)

type PoolingLayerState <: LayerState
  layer      :: PoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  etc        :: Any
end

function setup_etc(backend::CPUBackend, layer::PoolingLayer, inputs,
    pooled_width, pooled_height)

  if isa(layer.pooling, Pooling.Max)
    masks = Array(Array, length(inputs))
    for i = 1:length(inputs)
      masks[i] = Array(Csize_t, pooled_width, pooled_height,
          get_chann(inputs[i]), get_num(inputs[i]))
    end
    etc = masks
  else
    etc = nothing
  end
  return etc
end

function setup(backend::Backend, layer::PoolingLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  width    = get_width(inputs[1])
  height   = get_height(inputs[1])

  pooled_width  = int(ceil(float(width +2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]))+1
  pooled_height = int(ceil(float(height+2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]))+1

  # make sure the last pooling is not purely pooling padded area
  if layer.pad[1] > 0 || layer.pad[2] > 0
    if ((pooled_height-1) * layer.stride[2] >= height + layer.pad[2])
      pooled_height -= 1
    end
    if ((pooled_width-1) * layer.stride[1] >= width + layer.pad[1])
      pooled_width -= 1
    end
  end

  dtype = eltype(inputs[1])
  blobs = Array(Blob, length(inputs))
  blobs_diff = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    blobs[i] = make_blob(backend,dtype,
        (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
    if isa(diffs[i], NullBlob)
      blobs_diff[i] = NullBlob() # don't need back propagation unless bottom layer want
    else
      blobs_diff[i] = make_blob(backend,dtype,
          (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
    end
  end

  etc = setup_etc(backend, layer, inputs, pooled_width, pooled_height)
  state = PoolingLayerState(layer, blobs, blobs_diff, etc)
end
function shutdown(backend::CPUBackend, state::PoolingLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
end


function forward(backend::CPUBackend, state::PoolingLayerState, inputs::Vector{Blob})
  forward(backend, state.layer.pooling, state, inputs)
end

function forward(backend::CPUBackend, pool::StdPoolingFunction, state::PoolingLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data

    if isa(pool, Pooling.Max)
      max_pooling_forward(input, output, state.etc[i], state.layer)
    elseif isa(pool, Pooling.Mean)
      mean_pooling_forward(input, output, state.layer)
    else
      error("Pooling for $pool not implemented yet")
    end
  end
end

function backward(backend::CPUBackend, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(backend, state.layer.pooling, state, inputs, diffs)
end

function backward(backend::CPUBackend, pool::StdPoolingFunction, state::PoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      if isa(pool, Pooling.Max)
        max_pooling_backward(diff.data, state.blobs_diff[i].data, state.etc[i], state.layer)
      elseif isa(pool, Pooling.Mean)
        mean_pooling_backward(diff.data, state.blobs_diff[i].data, state.layer)
      else
        error("Pooling for $pool not implemented yet")
      end
    else
      continue # nothing to do if not propagating back
    end
  end
end

