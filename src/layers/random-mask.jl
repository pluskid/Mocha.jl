@defstruct RandomMaskLayer Layer (
  name :: AbstractString = "random-mask",
  (ratio :: AbstractFloat = 0.5, 0 < ratio < 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0)
)
@characterize_layer(RandomMaskLayer,
  can_do_bp => true,
  is_inplace => true,
)

type RandomMaskLayerState <: LayerState
  layer      :: RandomMaskLayer

  dropouts   :: Vector{DropoutLayerState}
end

function setup(backend::Backend, layer::RandomMaskLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  dropouts = Array{DropoutLayerState}(length(inputs))
  for i = 1:length(inputs)
    dropout_layer = DropoutLayer(name="$(layer.name)-dropout-$i", auto_scale=false, ratio=layer.ratio,
        bottoms=Symbol[Symbol("$(layer.bottoms[i])-$i")])
    dropouts[i] = setup(backend, dropout_layer, Blob[inputs[i]], Blob[diffs[i]])
  end
  return RandomMaskLayerState(layer, dropouts)
end

function shutdown(backend::Backend, state::RandomMaskLayerState)
  for dropout_state in state.dropouts
    shutdown(backend, dropout_state)
  end
end

function forward(backend::Backend, state::RandomMaskLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    forward(backend, state.dropouts[i], Blob[inputs[i]])
  end
end

function backward(backend::Backend, state::RandomMaskLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(inputs)
    backward(backend, state.dropouts[i], Blob[inputs[i]], Blob[diffs[i]])
  end
end
