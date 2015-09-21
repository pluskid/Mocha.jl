# Note ReshapeLayer is NOT a UtilLayer because it computes backward (via some upper layers)
@defstruct ReshapeLayer Layer (
  name :: AbstractString = "reshape",
  (tops :: Vector{Symbol} = [], length(tops) > 0),
  (bottoms :: Vector{Symbol} = [], length(bottoms) == length(tops)),
  (shape :: NTuple = (), eltype(shape) == Int && all(collect(shape) .> 0)),
)
@characterize_layer(ReshapeLayer,
  can_do_bp => true, # back-propagate via upper layers
)

type ReshapeLayerState <: LayerState
  layer      :: ReshapeLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::ReshapeLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs = map(inputs) do blob
    reshape_blob(backend, blob, layer.shape..., get_num(blob))
  end
  blobs_diff = map(diffs) do blob
    if isa(blob, NullBlob)
      NullBlob()
    else
      reshape_blob(backend, blob, layer.shape..., get_num(blob))
    end
  end

  return ReshapeLayerState(layer, blobs, blobs_diff)
end

function forward(backend::Backend, state::ReshapeLayerState, inputs::Vector{Blob})
end
function backward(backend::Backend, state::ReshapeLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end
function shutdown(backend::Backend, state::ReshapeLayerState)
end
