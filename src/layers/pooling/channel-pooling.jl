################################################################################
# Pooling in channels
################################################################################
function max_channel_pooling_forward{T}(input::Array{T,3}, output::Array{T,3}, mask::Array{Csize_t,3}, layer)
  spatial_dim, channels, num = size(input)
  pooled_chann = size(output, 2)

  for n = 1:num
    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)

      for s = 1:spatial_dim
        @inbounds output[s,pc,n] = input[s,cstart,n]
        @inbounds mask[s,pc,n] = cstart
      end

      for c = cstart+1:cend
        for s = 1:spatial_dim
          @inbounds maxval = output[s,pc,n]
          @inbounds val = input[s,c,n]
          if val > maxval
            @inbounds output[s,pc,n] = val
            @inbounds mask[s,pc,n] = c
          end
        end
      end
    end
  end
end

function mean_channel_pooling_forward{T}(input::Array{T,3}, output::Array{T,3}, integral::Array{T}, layer)
  spatial_dim_T, channels, num = size(input)
  pooled_chann = size(output, 2)
  one = convert(T, 1)
  neg_one = convert(T, -1)
  scale = 1/convert(T, layer.kernel)

  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = pointer(input) + fea_dim*(n-1)
    output_ptr = pointer(output) + output_fea_dim*(n-1)
    integral_ptr = pointer(integral)

    # compute integral image
    BLAS.blascopy!(spatial_dim_T, input_ptr, 1, integral_ptr, 1)
    for c = 2:channels
      BLAS.blascopy!(spatial_dim_T, input_ptr + (c-1)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
      BLAS.axpy!(spatial_dim_T, one, integral_ptr + (c-2)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
    end

    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)

      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      BLAS.blascopy!(spatial_dim_T, integral_ptr + (cend-1)*spatial_dim, 1,
          output_ptr_pc, 1)
      if cstart > 1
        BLAS.axpy!(spatial_dim_T, neg_one, integral_ptr + (cstart-2)*spatial_dim, 1,
            output_ptr_pc, 1)
      end
      BLAS.scal!(spatial_dim_T, scale, output_ptr_pc, 1)
    end
  end
end

function max_channel_pooling_backward{T}(input::Array{T,3}, output::Array{T,3}, mask::Array{Csize_t,3}, layer)
  spatial_dim, channels, num = size(input)
  pooled_chann = size(output, 2)

  fill!(input, 0)
  for n = 1:num
    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)

      for s = 1:spatial_dim
        @inbounds input[s,mask[s,pc,n],n] += output[s,pc,n]
      end
    end
  end
end

function mean_channel_pooling_backward{T}(input::Array{T,3}, output::Array{T,3}, layer)
  spatial_dim_T, channels, num = size(input)
  pooled_chann = size(output, 2)
  scale = 1/convert(T, layer.kernel)

  fill!(input, 0)

  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = pointer(input) + fea_dim*(n-1)
    output_ptr = pointer(output) + output_fea_dim*(n-1)

    for pc = 1:pooled_chann
      cstart = (pc-1)*layer.stride - layer.pad[1] + 1
      cend   = min(cstart + layer.kernel - 1, channels)
      cstart = max(1, cstart)
      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      for c = cstart:cend
        BLAS.axpy!(spatial_dim_T, scale, output_ptr_pc, 1,
            input_ptr + (c-1)*spatial_dim, 1)
      end
    end
  end
end

