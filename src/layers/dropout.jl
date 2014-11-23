@defstruct DropoutLayer InplaceLayer (
  name :: String = "dropout",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 1),
  (ratio :: FloatingPoint = 0.5, 0 < ratio < 1)
)

type DropoutLayerState{T} <: LayerState
  layer      :: DropoutLayer
  rand_vals  :: Blob

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
  rand_vals = make_blob(sys.backend, data_type, size(inputs[1]))

  etc = setup_etc(sys, layer, inputs)
  return DropoutLayerState(layer, rand_vals,
      convert(data_type, layer.ratio), convert(data_type, 1.0/(1-layer.ratio)), etc)
end
function destroy_etc(sys::System{CPUBackend}, state::DropoutLayerState)
  # do nothing
end
function shutdown(sys::System, state::DropoutLayerState)
  destroy(state.rand_vals)
  destroy_etc(sys, state)
end

function dropout_forward{T}(input::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(input)
  @simd for i = 1:len
    @inbounds input[i] = input[i] * (rand_vals[i] > ratio) * scale
  end
end
function forward(sys::System{CPUBackend}, state::DropoutLayerState, inputs::Vector{Blob})
  rand!(state.rand_vals.data)
  dropout_forward(inputs[1].data, state.rand_vals.data, state.ratio, state.scale)
end

function dropout_backward{T}(grad::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(grad)
  @simd for i = 1:len
    @inbounds grad[i] = grad[i] * (rand_vals[i] > ratio) * scale
  end
end
function backward(sys::System{CPUBackend}, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    dropout_backward(diffs[1].data, state.rand_vals.data, state.ratio, state.scale)
  end
end

