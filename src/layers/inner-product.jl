@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) >= 1),
  (bottoms :: Vector{String} = String[], length(bottoms) == length(tops)),
  weight_init :: Initializer = XavierInitializer(),
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
      for i = 1:length(inputs)
        blobs[i] = CPUBlob(data_type, out_dim, nums)
        blobs_diff[i] = CPUBlob(data_type, out_dim, nums)
      end

      state = new(layer, blobs, blobs_diff)
      state.W  = CPUBlob(data_type, fea_size, out_dim)
      state.∇W = CPUBlob(data_type, fea_size, out_dim)
      state.b  = CPUBlob(data_type, out_dim)
      state.∇b = CPUBlob(data_type, out_dim)

      state.bias_multiplier = CPUBlob(data_type, nums)
      fill!(state.bias_multiplier, 1)
    elseif isa(sys.backend, CuDNNBackend)
      for i = 1:length(inputs)
        blobs[i] = cudnn_make_tensor_blob(data_type, out_dim, nums)
        blobs_diff[i] = cudnn_make_tensor_blob(data_type, out_dim, nums)
      end

      state = new(layer, blobs, blobs_diff)
      state.W  = cudnn_make_tensor_blob(data_type, fea_size, out_dim)
      state.∇W = cudnn_make_tensor_blob(data_type, fea_size, out_dim)
      state.b  = cudnn_make_tensor_blob(data_type, out_dim)
      state.∇b = cudnn_make_tensor_blob(data_type, out_dim)

      state.bias_multiplier = cudnn_make_tensor_blob(data_type, nums)
      fill!(state.bias_multiplier, 1)
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
  M = size(state.W, 4)   # target dim
  N = size(inputs[1], 4) # batch size
  K = size(state.W, 3)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    # output = W^T * X
    BLAS.gemm!('T', 'N', convert(dtype, 1), reshape(state.W.data, (K,M)),
                reshape(input.data, (K,N)), convert(dtype, 0), reshape(output.data, (M,N)))
    # output += bias
    BLAS.gemm!('N', 'N', convert(dtype, 1), reshape(state.b.data, (M,1)),
                reshape(state.bias_multiplier.data, (1,N)), convert(dtype, 1), reshape(output.data, (M,N)))
  end
end

function backward(sys::System{CPUBackend}, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  target_dim = size(state.W, 4)
  source_dim = size(state.W, 3)
  batch_size = size(inputs[1], 4)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = convert(data_type, 0)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    ∂f_∂o = state.blobs_diff[i]
    BLAS.gemm!('N', 'T', convert(data_type, 1), reshape(input.data, (source_dim, batch_size)),
               reshape(∂f_∂o.data, (target_dim, batch_size)), zero_and_then_one,
               reshape(state.∇W.data, (source_dim, target_dim)))

    # ∂f/∂b = sum(∂f/∂o, 2)
    BLAS.gemm!('N', 'N', convert(data_type, 1), reshape(∂f_∂o.data, (target_dim, batch_size)),
               reshape(state.bias_multiplier.data, (batch_size, 1)), zero_and_then_one,
               reshape(state.∇b.data, (target_dim, 1)))

    zero_and_then_one = convert(data_type, 1)

    # if back propagate down
    if isa(diffs[i], CPUBlob)
      # ∂f/∂x = W * [∂f/∂o]
      BLAS.gemm!('N', 'N', convert(data_type, 1), reshape(state.W.data, (source_dim, target_dim)),
                 reshape(∂f_∂o.data, (target_dim, batch_size)), convert(data_type, 0),
                 reshape(diffs[i].data, (source_dim, batch_size)))
    end
  end
end


function forward(sys::System{CuDNNBackend}, state::InnerProductLayerState, inputs::Vector{Blob})
  M = size(state.W, 4)   # target dim
  N = size(inputs[1], 4) # batch size
  K = size(state.W, 3)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    # output = W^T * X
    CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_T, CuBLAS.OP_N, M, N, K, convert(dtype, 1),
                state.W.ptr, K, input.ptr, K, convert(dtype, 0), output.ptr, M)
    # output += bias
    CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, M, N, 1, convert(dtype, 1),
                state.b.ptr, M, state.bias_multiplier.ptr, 1, convert(dtype, 1), output.ptr, M)
  end
end

function backward(sys::System{CuDNNBackend}, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  target_dim = size(state.W, 4)
  source_dim = size(state.W, 3)
  batch_size = size(inputs[1], 4)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = convert(data_type, 0)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    ∂f_∂o = state.blobs_diff[i]
    CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_T, source_dim, target_dim, batch_size,
        convert(data_type, 1), input.ptr, source_dim, ∂f_∂o.ptr, target_dim, zero_and_then_one, state.∇W.ptr, source_dim)

    # ∂f/∂b = sum(∂f/∂o, 2)
    CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, target_dim, 1, batch_size,
        convert(data_type, 1), ∂f_∂o.ptr, target_dim, state.bias_multiplier.ptr, batch_size, zero_and_then_one, state.∇b.ptr, target_dim)

    zero_and_then_one = convert(data_type, 1)

    # if back propagate down
    if isa(diffs[i], CuTensorBlob)
      # ∂f/∂x = W * [∂f/∂o]
      CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, source_dim, batch_size, target_dim,
          convert(data_type, 1), state.W.ptr, source_dim, ∂f_∂o.ptr, target_dim, convert(data_type, 0), diffs[i].ptr, source_dim)
    end
  end
end
