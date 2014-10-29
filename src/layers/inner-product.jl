@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) == 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == 1),
  weight_init :: Initializer = ConstantInitializer(0),
  bias_init :: Initializer = ConstantInitializer(0),
  weight_regu :: Regularizer = L2Regu(1),
  bias_regu :: Regularizer = NoRegu(),
  neuron :: ActivationFunction = Neurons.Identity()
)

type InnerProductLayerState <: LayerState
  layer      :: InnerProductLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  W  :: Blob
  ∇W :: Blob
  b  :: Blob
  ∇b :: Blob

  InnerProductLayerState(sys::System, layer::InnerProductLayer, inputs::Vector{Blob}) = begin
    @assert length(inputs) == 1
    input = inputs[1]
    dims = size(input.data)
    left_dim = dims[1]
    mid_dim = dims[2:end]
    right_dim = layer.output_dim

    out_dim = (left_dim, right_dim)
    data_type = eltype(input.data)

    if isa(sys.backend, CPU)
      blobs = Blob[CPUBlob(Array(data_type, out_dim))]
      blobs_diff = Blob[CPUBlob(Array(data_type, out_dim))]
      state = new(layer, blobs, blobs_diff)
      state.W  = CPUBlob(Array(data_type, (prod(mid_dim), right_dim)))
      state.∇W = CPUBlob(Array(data_type, (prod(mid_dim), right_dim)))
      state.b  = CPUBlob(Array(data_type, (right_dim)))
      state.∇b = CPUBlob(Array(data_type, (right_dim)))
    else
      error("Backend $(sys.backend) not supported")
    end

    state.parameters = [Parameter(state.W, state.∇W, layer.weight_init, layer.weight_regu),
                        Parameter(state.b, state.∇b, layer.bias_init, layer.bias_regu)]

    return state
  end
end

function setup(sys::System, layer::InnerProductLayer, inputs::Vector{Blob})
  state = InnerProductLayerState(sys, layer, inputs)
  return state
end

function forward(sys::System{CPU}, state::InnerProductLayerState, inputs::Vector{Blob})
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

function backward(sys::System{CPU}, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  input = inputs[1]
  inner_dim = prod(size(input.data)[2:end])
  num = size(input.data,1)

  # Gradient w.r.t. parameters
  X = reshape(input.data, num, inner_dim)
  D = state.blobs_diff[1].data

  # ∇W = X' * D / N
  one = convert(eltype(X),1)
  zero = convert(eltype(X),0)
  BLAS.gemm!('T', 'N', one, X, D, zero, state.∇W.data)

  state.∇b.data[:] = sum(D,1)

  # Back propagate gradient w.r.t. input
  if isa(diffs[1], CPUBlob)
    # similar to ∇W
    BLAS.gemm!('N', 'T', one, D, state.W.data, zero, reshape(diffs[1].data,num,inner_dim))
  end
end
