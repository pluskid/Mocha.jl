function forward(backend::GPUBackend, state::TiedInnerProductLayerState, inputs::Vector{Blob})
  recon_dim  = size(state.W, 1)
  hidden_dim = size(state.W, 2)
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    N = get_num(input)   # batch size
    output = state.blobs[i]
    # output = (W^T)^T * X
    CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, recon_dim, N, hidden_dim, one(dtype),
                get_ptr(state.W), recon_dim, get_ptr(input), hidden_dim, zero(dtype), get_ptr(output), recon_dim)
    # output += bias
    CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, recon_dim, N, 1, one(dtype),
                get_ptr(state.b), recon_dim, get_ptr(state.bias_multipliers[i]), 1, one(dtype), get_ptr(output), recon_dim)
  end
end

function backward(backend::GPUBackend, state::TiedInnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  recon_dim  = size(state.W, 1)
  hidden_dim = size(state.W, 2)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = zero(data_type)

  for i = 1:length(inputs)
    input = inputs[i]
    batch_size = get_num(input)
    ∂f_∂o = state.blobs_diff[i]

    if !state.frozen
      # ∂f/∂b = sum(∂f/∂o, 2)
      CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, recon_dim, 1, batch_size,
          convert(data_type, 1), get_ptr(∂f_∂o), recon_dim, get_ptr(state.bias_multipliers[i]), batch_size, zero_and_then_one, get_ptr(state.∇b), recon_dim)
    end

    zero_and_then_one = one(data_type)

    # if back propagate down
    if isa(diffs[i], CuTensorBlob)
      # ∂f/∂x = W^T * [∂f/∂o]
      CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_T, CuBLAS.OP_N, hidden_dim, batch_size, recon_dim,
          convert(data_type, 1), get_ptr(state.W), recon_dim, get_ptr(∂f_∂o), recon_dim, convert(data_type, 0), get_ptr(diffs[i]), hidden_dim)
    end
  end
end
