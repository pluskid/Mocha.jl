############################################################
# NOTE: dropout layer only takes one input blob. We
# intentionally make this restriction because as an
# in-place layer, if has to be put directly on top
# of the layer that is producing the input blob for
# drop-out layer. If drop-out layer depends on multiple
# blobs coming from different layers, it might be
# difficult or even impossible to decide a correct
# layer layout.
############################################################
@defstruct DropoutLayer Layer (
  name :: AbstractString = "dropout",
  auto_scale :: Bool = true,
  (ratio :: AbstractFloat = 0.5, 0 < ratio < 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 1),
)
@characterize_layer(DropoutLayer,
  can_do_bp  => true,
  is_inplace => true
)

type DropoutLayerState{T} <: LayerState
  layer      :: DropoutLayer
  rand_vals  :: Blob

  # for convenience
  ratio      :: T   # layer.ratio
  scale      :: T   # 1 / (1 - layer.ratio)

  etc        :: Any
end

function setup_etc(backend::CPUBackend, layer::DropoutLayer, inputs::Vector{Blob})
  return nothing
end
function setup(backend::Backend, layer::DropoutLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  rand_vals = make_blob(backend, data_type, size(inputs[1]))

  scale = layer.auto_scale ? 1.0/(1-layer.ratio) : 1

  etc = setup_etc(backend, layer, inputs)
  return DropoutLayerState(layer, rand_vals,
      convert(data_type, layer.ratio), convert(data_type, scale), etc)
end
function destroy_etc(backend::CPUBackend, state::DropoutLayerState)
  # do nothing
end
function shutdown(backend::Backend, state::DropoutLayerState)
  destroy(state.rand_vals)
  destroy_etc(backend, state)
end

function dropout_forward{T}(input::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(input)
  @simd for i = 1:len
    @inbounds input[i] = input[i] * (rand_vals[i] > ratio) * scale
  end
end
function forward(backend::CPUBackend, state::DropoutLayerState, inputs::Vector{Blob})
  rand!(state.rand_vals.data)
  dropout_forward(inputs[1].data, state.rand_vals.data, state.ratio, state.scale)
end

function dropout_backward{T}(grad::Array{T}, rand_vals::Array{T}, ratio::T, scale::T)
  len = length(grad)
  @simd for i = 1:len
    @inbounds grad[i] = grad[i] * (rand_vals[i] > ratio) * scale
  end
end
function backward(backend::CPUBackend, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    dropout_backward(diffs[1].data, state.rand_vals.data, state.ratio, state.scale)
  end
end

