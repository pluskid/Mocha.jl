function setup_etc(sys::System{CuDNNBackend}, layer::ChannelPoolingLayer, inputs, pooled_chann)
  if isa(layer.pooling, Pooling.Max)
    masks = Array(CuPtr, length(inputs))
    for i = 1:length(inputs)
      masks[i] = CUDA.cualloc(Csize_t, get_width(inputs[i]) * get_height(inputs[i]) *
          pooled_chann * get_num(inputs[i]))
    end
    etc = masks
  elseif isa(layer.pooling, Pooling.Mean)
    integrals = Array(CuPtr, length(inputs))
    for i = 1:length(inputs)
      integrals[i] = CUDA.cualloc(eltype(inputs[i]), get_width(inputs[i]) * get_height(inputs[i]) *
          get_chann(inputs[i]))
    end
    etc = integrals
  else
    etc = nothing
  end
  return etc
end
function shutdown_etc(sys::System{CuDNNBackend}, state::ChannelPoolingLayerState)
  if isa(state.layer.pooling, Pooling.Max)
    map(CUDA.free, state.etc)
  elseif isa(state.layer.pooling, Pooling.Mean)
    map(CUDA.free, state.etc)
  else
    error("Unknown pooling $(state.layer.pooling)")
  end
end

function forward(sys::System{CuDNNBackend}, state::ChannelPoolingLayerState, inputs::Vector{Blob})
  forward(sys, state.layer.pooling, state, inputs)
end
function forward(sys::System{CuDNNBackend}, pool::StdPoolingFunction,
    state::ChannelPoolingLayerState, inputs::Vector{Blob})

  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    if isa(pool, Pooling.Max)
      cuda_max_channel_pooling_forward(sys, input, output, state.etc[i], state.layer)
    elseif isa(pool, Pooling.Mean)
      cuda_mean_channel_pooling_forward(sys, input, output, state.etc[i], state.layer)
    else
      error("Pooling for $pool not implemented yet")
    end
  end
end

function backward(sys::System{CuDNNBackend}, state::ChannelPoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(sys, state.layer.pooling, state, inputs, diffs)
end

function backward(sys::System{CuDNNBackend}, pool::StdPoolingFunction, state::ChannelPoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      if isa(pool, Pooling.Max)
        cuda_max_channel_pooling_backward(sys, diff, state.blobs_diff[i], state.etc[i], state.layer)
      elseif isa(pool, Pooling.Mean)
        cuda_mean_channel_pooling_backward(sys, diff, state.blobs_diff[i], state.layer)
      else
        error("Pooling for $pool not implemented yet")
      end
    else
      continue # nothing to do if not propagating back
    end
  end
end

function cuda_mean_channel_pooling_forward{T}(sys::System{CuDNNBackend}, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, integral::CuPtr, layer)

  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)
  one = convert(T, 1)
  neg_one = convert(T, -1)
  scale = convert(T, 1.0/layer.kernel)

  spatial_dim_T = width*height
  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = convert(Ptr{T}, input.ptr.p) + fea_dim*(n-1)
    output_ptr = convert(Ptr{T}, output.ptr.p) + output_fea_dim*(n-1)
    integral_ptr = convert(Ptr{T}, integral.p)

    # compute integral image
    CuBLAS.copy(sys.backend.cublas_ctx, T, spatial_dim_T, input_ptr, 1, integral_ptr, 1)
    for c = 2:channels
      CuBLAS.copy(sys.backend.cublas_ctx, T, spatial_dim_T, input_ptr + (c-1)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
      CuBLAS.axpy(sys.backend.cublas_ctx, spatial_dim_T, one, integral_ptr + (c-2)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
    end

    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)

      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      CuBLAS.copy(sys.backend.cublas_ctx, T, spatial_dim_T, integral_ptr + (cend-1)*spatial_dim, 1,
          output_ptr_pc, 1)
      if cstart > 1
        CuBLAS.axpy(sys.backend.cublas_ctx, spatial_dim_T, neg_one,
            integral_ptr + (cstart-2)*spatial_dim, 1, output_ptr_pc, 1)
      end
      CuBLAS.scal(sys.backend.cublas_ctx, spatial_dim_T, scale, output_ptr_pc, 1)
    end
  end
end

function cuda_mean_channel_pooling_backward{T}(sys::System{CuDNNBackend}, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, layer)

  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)
  scale = 1/convert(T, layer.kernel)

  fill!(input, 0)

  spatial_dim_T = width*height
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
        CuBLAS.axpy(sys.backend.cublas_ctx, spatial_dim_T, scale, output_ptr_pc, 1,
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
function cuda_max_channel_pooling_forward{T}(sys::System{CuDNNBackend}, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, mask::CuPtr, layer)

  width, height, channels, num = size(input)
  sp_dim = width*height
  pooled_chann = get_chann(output)

  cuda_dim = cuda_geometry_max_chann_pool(sp_dim, num);
  if T == Float32
    kernel = sys.backend.mocha.max_channel_pooling_forward_float
  elseif T == Float64
    kernel = sys.backend.mocha.max_channel_pooling_forward_double
  else
    error("Unsupported data type for channel pooling: $T")
  end

  CUDA.launch(kernel, cuda_dim..., (input.ptr.p, output.ptr.p, mask.p, sp_dim, channels, num,
      pooled_chann, layer.kernel, layer.stride, layer.pad[1]))
end

function cuda_max_channel_pooling_backward{T}(sys::System{CuDNNBackend}, input::CuTensorBlob{T},
    output::CuTensorBlob{T}, mask::CuPtr, layer)

  width, height, channels, num = size(input)
  sp_dim = width*height
  pooled_chann = get_chann(output)

  cuda_dim = cuda_geometry_max_chann_pool(sp_dim, num);
  if T == Float32
    kernel = sys.backend.mocha.max_channel_pooling_backward_float
  elseif T == Float64
    kernel = sys.backend.mocha.max_channel_pooling_backward_double
  else
    error("Unsupported data type for channel pooling: $T")
  end
  erase!(input)

  CUDA.launch(kernel, cuda_dim..., (input.ptr.p, output.ptr.p, mask.p, sp_dim, channels, num,
      pooled_chann, layer.kernel, layer.stride, layer.pad[1]))
end

