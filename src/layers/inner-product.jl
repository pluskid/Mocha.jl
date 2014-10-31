@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) >= 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == length(tops)),
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

  # a all-1 vector used in gemm to help bias calculation
  bias_multiplier :: Blob

  InnerProductLayerState(sys::System, layer::InnerProductLayer, inputs::Vector{Blob}) = begin
    dims = size(inputs[1])
    nums = dims[4]
    fea_dim = dims[1:3]
    fea_size = prod(fea_dim)
    out_dim = layer.output_dim

    data_type = eltype(inputs[1])
    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))

    if isa(sys.backend, CPUBackend)
      # TODO: correct this
      #blobs = Blob[CPUBlob(Array(data_type, out_dim))]
      #blobs_diff = Blob[CPUBlob(Array(data_type, out_dim))]
      #state = new(layer, blobs, blobs_diff)
      #state.W  = CPUBlob(Array(data_type, (prod(mid_dim), right_dim)))
      #state.∇W = CPUBlob(Array(data_type, (prod(mid_dim), right_dim)))
      #state.b  = CPUBlob(Array(data_type, (right_dim)))
      #state.∇b = CPUBlob(Array(data_type, (right_dim)))
    elseif isa(sys.backend, CuDNNBackend)
      for i = 1:length(inputs)
        blobs[i] = cudnn_make_tensor_blob(data_type, fea_size, nums)
        blobs_diff[i] = cudnn_make_tensor_blob(data_type, fea_size, nums)
      end

      state = new(layer, blobs, blobs_diff)
      state.W  = cudnn_make_pod_blob(data_type, fea_size, out_dim)
      state.∇W = cudnn_make_pod_blob(data_type, fea_size, out_dim)
      state.b  = cudnn_make_pod_blob(data_type, out_dim)
      state.∇b = cudnn_make_pod_blob(data_type, out_dim)

      state.bias_multiplier = cudnn_make_pod_blob(data_type, nums)
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

function forward(sys::System{CPUBackend}, state::InnerProductLayerState, inputs::Vector{Blob})
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

function backward(sys::System{CPUBackend}, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
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


function forward(sys::System{CuDNNBackend}, state::InnerProductLayerState, inputs::Vector{blob})
  M = size(state.W, 4)   # target dim
  N = size(inputs[1], 4) # batch size
  K = size(state.W, 3)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    # output = W^T * X
    CuBLAS.gemm(sys.backend.cublas_ctx, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, convert(dtype, 1),
                state.W.ptr, K, input.ptr, K, convert(dtype, 0), output.ptr, M)
    # output += bias
    CuBLAS.gemm(sys.backend.cublas_ctx, CUBLAS_OP_T, CUBLAS_OP_N, M, N, 1, convert(dtype, 1),
                state.b.ptr, 1, state.bias_multiplier.ptr, 1, convert(dtype, 1), output.ptr, M)
  end
end
