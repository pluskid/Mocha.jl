export reset_outputs

@defstruct MemoryOutputLayer Layer (
  name :: AbstractString = "memory-output",
  (bottoms :: Vector{Symbol} = [], length(bottoms) > 0)
)
@characterize_layer(MemoryOutputLayer,
  is_sink => true
)

type MemoryOutputLayerState <: LayerState
  layer   :: MemoryOutputLayer
  outputs :: Vector{Vector{Array}}
end

function reset_outputs(state::MemoryOutputLayerState)
  for i = 1:length(state.outputs)
    state.outputs[i] = Array[]
  end
end

function setup(backend::Backend, layer::MemoryOutputLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  outputs = Array{Vector{Array}}(length(inputs))
  for i = 1:length(inputs)
    outputs[i] = Array[]
  end

  return MemoryOutputLayerState(layer, outputs)
end

function forward(backend::Backend, state::MemoryOutputLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    push!(state.outputs[i], to_array(inputs[i]))
  end
end

function backward(backend::Backend, state::MemoryOutputLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

function shutdown(backend::Backend, state::MemoryOutputLayerState)
end
