@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) == 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == 1)
)

type InnerProductLayerState <: LayerState
  layer :: InnerProductLayer
  blobs :: Vector{Blob}

  W :: Blob
  b :: Blob

  ∇W :: Blob
  ∇b :: Blob

  InnerProductLayerState(layer::InnerProductLayer, inputs::Vector{Blob}) = begin
    @assert length(inputs) == 1
    input = inputs[1]
    in_dim = size(input.data)
    out_dim = (in_dim[1], layer.output_dim)

    data_type = eltype(input.data)
    state = new(layer, Blob[Blob(layer.tops[1], Array(data_type, out_dim))])
    state.W  = Blob("W", Array(data_type, tuple(in_dim[2:end]..., layer.output_dim)))
    state.∇W = Blob("∇W", Array(data_type, size(state.W.data)))
    state.b  = Blob("b", Array(data_type, layer.output_dim))
    state.∇b = Blob("∇b", Array(data_type, layer.output_dim))

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
  A = reshape(input.data, size(input.data,1), inner_dim)
  B = reshape(state.W.data, inner_dim, size(state.W.data)[end])
  C_blob = state.blobs[1]

  # C = A*B + C
  for i = 1:size(input.data,1)
    C_blob.data[i, :] = state.b.data
  end
  BLAS.gemm!('N', 'N', one, A, B, one, C_blob.data)
end
