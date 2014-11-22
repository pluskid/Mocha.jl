@defstruct DropoutLayer CompLayer (
  name :: String = "dropout",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (ratio :: FloatingPoint = 0.5, 0 < ratio < 1)
)

type DropoutLayerState{T} <: LayerState
  layer      :: DropoutLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
  rand_vals  :: Vector{Blob}

  # for convenience
  ratio      :: T   # layer.ratio
  scale      :: T   # 1 / (1 - layer.ratio)

  etc        :: Any
end

function setup_etc(sys::System{CPUBackend}, layer::DropoutLayer, inputs::Vector{Blob})
  return nothing
end
function setup(sys::System, layer::DropoutLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  blobs = Blob[make_blob(sys.backend, data_type, size(x)) for x in inputs]
  blobs_diff = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    if isa(diffs[i], NullBlob)
      blobs_diff[i] = NullBlob()
    else
      blobs_diff[i] = make_blob(sys.backend, data_type, size(diffs[i]))
    end
  end
  rand_vals = Blob[make_blob(sys.backend, data_type, size(x)) for x in inputs]

  etc = setup_etc(sys, layer, inputs)
  return DropoutLayerState(layer, blobs, blobs_diff, rand_vals,
      convert(data_type, layer.ratio), convert(data_type, 1.0/(1-layer.ratio)), etc)
end

function dropout_forward{T}(input::Array{T}, output::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(input)
  @simd for i = 1:len
    @inbounds output[i] = input[i] * (rand_vals[i] > ratio) * scale
  end
end
function forward(sys::System{CPUBackend}, state::DropoutLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    rand!(state.rand_vals[i].data)
    dropout_forward(inputs[i].data, state.blobs[i].data, state.rand_vals[i].data, state.ratio, state.scale)
  end
end

function dropout_backward{T}(grad::Array{T}, top_diff::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(grad)
  @simd for i = 1:len
    @inbounds grad[i] = top_diff[i] * (rand_vals[i] > ratio) * scale
  end
end
function backward(sys::System{CPUBackend}, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      dropout_backward(diffs[i].data, state.blobs_diff[i].data, state.rand_vals[i].data, state.ratio, state.scale)
    end
  end
end

function destroy_etc(sys::System{CPUBackend}, state::DropoutLayerState)
  # do nothing
end
function shutdown(sys::System, state::DropoutLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  map(destroy, state.rand_vals)
  destroy_etc(sys, state)
end

