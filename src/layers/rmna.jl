############################################################
# NOTE: rmna layer only takes one input blob.
# implementation modified from dropout layer
############################################################
@defstruct RmNaLayer Layer (
  name :: AbstractString = "rmna",
  NAval :: AbstractFloat = -9999.0,
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 1)
)
@characterize_layer(RmNaLayer,
  can_do_bp  => true,
  is_inplace => true
)

type RmNaLayerState{T} <: LayerState
  layer     :: RmNaLayer
  is_na     :: Blob

  NAval      :: T

  etc        :: Any
end

function setup_etc(backend::CPUBackend, layer::RmNaLayer, inputs::Vector{Blob})
  return nothing
end
function setup(backend::Backend, layer::RmNaLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  is_na = make_blob(backend, data_type, size(inputs[1]))

  etc = setup_etc(backend, layer, inputs)
  return RmNaLayerState(layer, is_na, layer.NAval, etc)
end
function destroy_etc(backend::CPUBackend, state::RmNaLayerState)
  # do nothing
end
function shutdown(backend::Backend, state::RmNaLayerState)
  destroy(state.is_na)
  destroy_etc(backend, state)
end

function forward(backend::CPUBackend, state::RmNaLayerState, inputs::Vector{Blob})
  is_na!(state.is_na.data, inputs[1].data, state.NAval)
  rmna!(inputs[1].data, inputs[1].data, state.NAval)
end

# TODO: do we need to save the NA state of the layer, wouldn't it
# be much simpler to just call rmna! in forward and backwards pass?
# in other words, may the layer state change? i would guess that
# this is not possible.
#=function rmna_backward{T}(grad::Array{T}, is_na::Array{T})
  Vec.mul!(grad.data, is_na.data)
end=#
function backward(backend::CPUBackend, state::RmNaLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    Vec.mul!(diffs[1].data, state.is_na.data)
  end
end

#=
function rmna_forward{T}(input::Array{T}, is_na::Array{T})
  len = length(input)
  @simd for i = 1:len
    @inbounds input[i] = input[i] * is_na[i]
  end
end=#
function is_na!{T,N}(X::Array{T,N}, Y::Array{T,N}, NAval::T)
  if isnan(NAval)
    @simd for i in eachindex(X)
      @inbounds X[i] = isnan(Y[i]) ? zero(NAval) : one(NAval)
    end
  else
    @simd for i in eachindex(X)
      @inbounds X[i] = Y[i] == NAval ? zero(NAval) : one(NAval)
    end
  end
end
