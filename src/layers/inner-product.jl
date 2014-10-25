@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) == 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == 1)
)

type InnerProductLayerState <: LayerState
  layer :: InnerProductLayer
  blobs :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  W :: Blob
  b :: Blob

  InnerProductLayerState(layer::InnerProductLayer, inputs::Vector{Blob}) = begin
    @assert length(inputs) == 1
    input = inputs[1]
    dims = size(input.data)
    left_dim = dims[1]
    mid_dim = dims[2:end]
    right_dim = layer.output_dim

    out_dim = (left_dim, right_dim)
    data_type = eltype(input.data)

    blobs = Blob[Blob(layer.tops[1], Array(data_type, out_dim))]
    blobs_diff = Blob[Blob(layer.tops[1], Array(data_type, out_dim))]
    state = new(layer, blobs, blobs_diff)
    state.W  = Blob("W", Array(data_type, (prod(mid_dim), right_dim)))
    state.b  = Blob("b", Array(data_type, (right_dim)))

    return state
  end
end

function setup(layer::InnerProductLayer, inputs::Vector{Blob})
  state = InnerProductLayerState(layer, inputs)
  return state
end

function forward(state::InnerProductLayerState, inputs::Vector{Blob})
  input = inputs[1]
  inner_dim = prod(size(input.data)[2:end])

  one = convert(eltype(input.data), 1)
  X = reshape(input.data, size(input.data,1), inner_dim)
  C_blob = state.blobs[1]

  # C = X*W + C
  for i = 1:size(input.data,1)
    C_blob.data[i, :] = state.b.data
  end
  BLAS.gemm!('N', 'N', one, X, state.W.data, one, C_blob.data)
end

function backward(state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # Update parameters

  # Back propagate
end
