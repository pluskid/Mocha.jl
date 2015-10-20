gemm(::Type{Float32}, args...) = clblas.clblasSgemm(args...)
gemm(::Type{Float64}, args...) = clblas.clblasDgemm(args...)

function forward(backend::OpenCLBackend, state::InnerProductLayerState, inputs::Vector{Blob})
  M = size(state.W, 2)   # target dim
  K = size(state.W, 1)   # source dim
  dtype = eltype(state.W)
#   gemm = f_gemm(dtype)

#   if dtype == Float32
#     gemm = clblas.clblasSgemm
#   elseif dtype == Float64
#     gemm = clblas.clblasDgemm
#   else
#     @error("Unsupported dtype: $dtype")
#   end

  for i = 1:length(inputs)
    input = inputs[i]
    N = get_num(input)   # batch size
    output = state.blobs[i]
    # output = W^T * X
    gemm(dtype, clblas.clblasColumnMajor, clblas.clblasTrans, clblas.clblasNoTrans, M, N, K, one(dtype),
         state.W.buffer.id, 0, K, input.buffer.id, 0, K, zero(dtype), output.buffer.id, 0, M,
         [backend.queue])
    # output += bias
    gemm(dtype, clblas.clblasColumnMajor, clblas.clblasNoTrans, clblas.clblasNoTrans, M, N, 1, one(dtype),
         state.b.buffer.id, 0, M, state.bias_multipliers[i].buffer.id, 0, 1, one(dtype), output.buffer.id,
         0, M, [backend.queue])
  end
end

function backward(backend::OpenCLBackend, state::InnerProductLayerState, inputs::Vector{Blob},
                  diffs::Vector{Blob})
  target_dim = size(state.W, 2)
  source_dim = size(state.W, 1)
  data_type  = eltype(state.W)
#   gemm = f_gemm(data_type)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = zero(data_type)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    batch_size = get_num(input)
    ∂f_∂o = state.blobs_diff[i]

    if !state.frozen
      gemm(data_type, clblas.clblasColumnMajor, clblas.clblasNoTrans, clblas.clblasTrans, source_dim, target_dim, batch_size,
           one(data_type), input.buffer.id, 0, source_dim, ∂f_∂o.buffer.id, 0, target_dim, zero_and_then_one, state.∇W.buffer.id, 0, source_dim,
           [backend.queue])

      # ∂f/∂b = sum(∂f/∂o, 2)
      gemm(data_type, clblas.clblasColumnMajor, clblas.clblasNoTrans, clblas.clblasNoTrans, target_dim, 1, batch_size,
           one(data_type), ∂f_∂o.buffer.id, 0, target_dim, state.bias_multipliers[i].buffer.id, 0, batch_size, zero_and_then_one, state.∇b.buffer.id, 0, target_dim,
           [backend.queue])
    end

    zero_and_then_one = one(data_type)

    # if back propagate down
    if isa(diffs[i], ClTensorBlob)
      # ∂f/∂x = W * [∂f/∂o]
      gemm(data_type, clblas.clblasColumnMajor, clblas.clblasNoTrans, clblas.clblasNoTrans, source_dim, batch_size, target_dim,
           one(data_type), state.W.buffer.id, 0, source_dim, ∂f_∂o.buffer.id, 0, target_dim, zero(data_type), diffs[i].buffer.id, 0, source_dim,
           [backend.queue])
    end
  end
end
