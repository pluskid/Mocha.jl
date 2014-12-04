# Note ReshapeLayer is NOT a UtilLayer because it computes backward (via some upper layers)
@defstruct ReshapeLayer CompLayer (
  name :: String = "reshape",
  (tops :: Vector{Symbol} = [], length(tops) > 0),
  (bottoms :: Vector{Symbol} = [], length(bottoms) == length(tops)),
  (width :: Int = 1, width > 0),
  (height :: Int = 1, height > 0),
  (channels :: Int = 1, channels > 0)
)

type ReshapeLayerState <: LayerState
  layer      :: ReshapeLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(sys::System, layer::ReshapeLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs = map(inputs) do blob
    reshape_blob(sys.backend, blob, layer.width, layer.height, layer.channels, get_num(blob))
  end
  blobs_diff = map(diffs) do blob
    if isa(blob, NullBlob)
      NullBlob()
    else
      reshape_blob(sys.backend, blob, layer.width, layer.height, layer.channels, get_num(blob))
    end
  end

  return ReshapeLayerState(layer, blobs, blobs_diff)
end

function forward(sys::System, state::ReshapeLayerState, inputs::Vector{Blob})
end
function backward(sys::System, state::ReshapeLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end
function shutdown(sys::System, state::ReshapeLayerState)
end
