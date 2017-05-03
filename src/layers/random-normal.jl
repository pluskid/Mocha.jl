
@defstruct RandomNormalLayer Layer (
  name :: AbstractString = "random-normal",
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (output_dims :: Vector{Int} = Int[], length(output_dims) >0),
  (eltype :: DataType = Float32),
  (batch_sizes :: Vector{Int} = Int[], (length(batch_sizes)==length(tops))),
)

@characterize_layer(RandomNormalLayer,
  is_source => true
)

type RandomNormalLayerState <: LayerState
    layer :: RandomNormalLayer
    blobs :: Vector{Blob}
    etc        :: Vector{Any}

    RandomNormalLayerState(backend::Backend, layer::RandomNormalLayer) = begin
      blobs = Array{Blob}(length(layer.tops))
      for i = 1:length(blobs)
        dims = tuple(layer.output_dims..., layer.batch_sizes[i])
        blobs[i] = make_blob(backend, layer.eltype, dims...)
    end
    new(layer, blobs, Any[])
  end
end

function setup_etc(backend::CPUBackend, layer::RandomNormalLayer)
  return Any[]
end

function destroy_etc(backend::CPUBackend, state::RandomNormalLayerState)
  # do nothing
end

function setup(backend::Backend, layer::RandomNormalLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  state = RandomNormalLayerState(backend, layer)
  state.etc = setup_etc(backend, layer)
  return state
end

function shutdown(backend::Backend, state::RandomNormalLayerState)
  map(destroy, state.blobs)
  destroy_etc(backend, state)
end

function forward(backend::CPUBackend, state::RandomNormalLayerState, inputs::Vector{Blob})
  for i = 1:length(state.blobs)
      randn!(state.blobs[i])
  end
end

function backward(backend::Backend, state::RandomNormalLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end
