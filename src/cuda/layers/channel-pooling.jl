function setup_etc(backend::GPUBackend, layer::ChannelPoolingLayer, inputs, blobs)
  if isa(layer.pooling, Pooling.Max)
    masks = Array(CuPtr, length(inputs))
    for i = 1:length(inputs)
      masks[i] = CUDA.cualloc(Csize_t, length(blobs[i]))
    end
    etc = masks
  elseif isa(layer.pooling, Pooling.Mean)
    integrals = Array(CuPtr, length(inputs))
    for i = 1:length(inputs)
      integrals[i] = CUDA.cualloc(eltype(inputs[i]), prod(size(inputs[i])[1:end-1]))
    end
    etc = integrals
  else
    etc = nothing
  end
  return etc
end
function shutdown_etc(backend::GPUBackend, state::ChannelPoolingLayerState)
  if isa(state.layer.pooling, Pooling.Max)
    map(CUDA.free, state.etc)
  elseif isa(state.layer.pooling, Pooling.Mean)
    map(CUDA.free, state.etc)
  else
    error("Unknown pooling $(state.layer.pooling)")
  end
end

function forward(backend::GPUBackend, state::ChannelPoolingLayerState, inputs::Vector{Blob})
  forward(backend, state.layer.pooling, state, inputs)
end
function forward(backend::GPUBackend, pool::StdPoolingFunction,
    state::ChannelPoolingLayerState, inputs::Vector{Blob})

  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    if isa(pool, Pooling.Max)
      cuda_max_channel_pooling_forward(backend, input, output, state.etc[i], state.layer, state.op_dims[i])
    elseif isa(pool, Pooling.Mean)
      cuda_mean_channel_pooling_forward(backend, input, output, state.etc[i], state.layer, state.op_dims[i])
    else
      error("Pooling for $pool not implemented yet")
    end
  end
end

function backward(backend::GPUBackend, state::ChannelPoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(backend, state.layer.pooling, state, inputs, diffs)
end

function backward(backend::GPUBackend, pool::StdPoolingFunction, state::ChannelPoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      if isa(pool, Pooling.Max)
        cuda_max_channel_pooling_backward(backend, diff, state.blobs_diff[i], state.etc[i], state.layer, state.op_dims[i])
      elseif isa(pool, Pooling.Mean)
        cuda_mean_channel_pooling_backward(backend, diff, state.blobs_diff[i], state.layer, state.op_dims[i])
      else
        error("Pooling for $pool not implemented yet")
      end
    else
      continue # nothing to do if not propagating back
    end
  end
end

function cuda_mean_channel_pooling_forward{T}(backend::GPUBackend, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, integral::CuPtr, layer, op_dim)

  spatial_dim_T, channels, num = split_dims(input, op_dim)
  pooled_chann = size(output, op_dim)
  one = convert(T, 1)
  neg_one = convert(T, -1)
  scale = convert(T, 1.0/layer.kernel)

  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = convert(Ptr{T}, input.ptr.p) + fea_dim*(n-1)
    output_ptr = convert(Ptr{T}, output.ptr.p) + output_fea_dim*(n-1)
    integral_ptr = convert(Ptr{T}, integral.p)

    # compute integral image
    CuBLAS.copy(backend.cublas_ctx, T, spatial_dim_T, input_ptr, 1, integral_ptr, 1)
    for c = 2:channels
      CuBLAS.copy(backend.cublas_ctx, T, spatial_dim_T, input_ptr + (c-1)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
      CuBLAS.axpy(backend.cublas_ctx, spatial_dim_T, one, integral_ptr + (c-2)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
    end

    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)

      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      CuBLAS.copy(backend.cublas_ctx, T, spatial_dim_T, integral_ptr + (cend-1)*spatial_dim, 1,
          output_ptr_pc, 1)
      if cstart > 1
        CuBLAS.axpy(backend.cublas_ctx, spatial_dim_T, neg_one,
            integral_ptr + (cstart-2)*spatial_dim, 1, output_ptr_pc, 1)
      end
      CuBLAS.scal(backend.cublas_ctx, spatial_dim_T, scale, output_ptr_pc, 1)
    end
  end
end

function cuda_mean_channel_pooling_backward{T}(backend::GPUBackend, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, layer, op_dim)

  spatial_dim_T, channels, num = split_dims(input, op_dim)
  pooled_chann = size(output, op_dim)
  scale = 1/convert(T, layer.kernel)

  fill!(input, 0)

  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = convert(Ptr{T}, input.ptr.p) + fea_dim*(n-1)
    output_ptr = convert(Ptr{T}, output.ptr.p) + output_fea_dim*(n-1)

    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)
      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      for c = cstart:cend
        CuBLAS.axpy(backend.cublas_ctx, spatial_dim_T, scale, output_ptr_pc, 1,
            input_ptr + (c-1)*spatial_dim, 1)
      end
    end
  end
end


function cuda_geometry_max_chann_pool(sp_dim::Int, num::Int)
  x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X));
  y_block = 1;
  z_block = int(ceil(float64(sp_dim)/CUDA.THREADS_PER_BLOCK_Z));
  return ((x_block,y_block,z_block),
          (CUDA.THREADS_PER_BLOCK_X,1,CUDA.THREADS_PER_BLOCK_Z))

end
function cuda_max_channel_pooling_forward{T}(backend::GPUBackend, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, mask::CuPtr, layer, op_dim)

  sp_dim, channels, num = split_dims(input, op_dim)
  pooled_chann = size(output, op_dim)

  cuda_dim = cuda_geometry_max_chann_pool(sp_dim, num);
  if T == Float32
    kernel = backend.mocha.max_channel_pooling_forward_float
  elseif T == Float64
    kernel = backend.mocha.max_channel_pooling_forward_double
  else
    error("Unsupported data type for channel pooling: $T")
  end

  CUDA.launch(kernel, cuda_dim..., (input.ptr.p, output.ptr.p, mask.p, sp_dim, channels, num,
      pooled_chann, layer.kernel, layer.stride, layer.pad[1]))
end

function cuda_max_channel_pooling_backward{T}(backend::GPUBackend, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, mask::CuPtr, layer, op_dim)

  sp_dim, channels, num = split_dims(input, op_dim)
  pooled_chann = size(output, op_dim)

  cuda_dim = cuda_geometry_max_chann_pool(sp_dim, num);
  if T == Float32
    kernel = backend.mocha.max_channel_pooling_backward_float
  elseif T == Float64
    kernel = backend.mocha.max_channel_pooling_backward_double
  else
    error("Unsupported data type for channel pooling: $T")
  end
  erase!(input)

  CUDA.launch(kernel, cuda_dim..., (input.ptr.p, output.ptr.p, mask.p, sp_dim, channels, num,
      pooled_chann, layer.kernel, layer.stride, layer.pad[1]))
end

