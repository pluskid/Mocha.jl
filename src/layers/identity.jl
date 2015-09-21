############################################################
# Identity Layer
############################################################
@defstruct IdentityLayer Layer (
  name :: AbstractString = "identity",
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)
@characterize_layer(IdentityLayer,
  can_do_bp => true
)

type IdentityLayerState <: LayerState
  layer      :: IdentityLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::IdentityLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs      = inputs[:] # shallow copy
  blobs_diff = diffs[:] # shallow_copy

  IdentityLayerState(layer, blobs, blobs_diff)
end
function shutdown(backend::Backend, state::IdentityLayerState)
end

function forward(backend::Backend, state::IdentityLayerState, inputs::Vector{Blob})
  # do nothing
end

function backward(backend::Backend, state::IdentityLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # do nothing
end
