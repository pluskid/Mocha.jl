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
