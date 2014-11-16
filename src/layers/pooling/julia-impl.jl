################################################################################
# Pooling in image dimension (width and height)
################################################################################
function max_pooling_forward{T}(input::Array{T}, output::Array{T}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*layer.stride[2] - layer.pad[2] + 1)
          wstart = max(1, (pw-1)*layer.stride[1] - layer.pad[1] + 1)
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)

          maxval = -Inf
          maxw = 0
          maxh = 0
          for w = wstart:wend
            for h = hstart:hend
              @inbounds val = input[w,h,c,n]
              if val > maxval
                maxval = val
                maxw = w
                maxh = h
              end
            end
          end
          @inbounds output[pw,ph,c,n] = maxval
          @inbounds mask[pw,ph,c,n] = (maxh-1) * width + maxw-1
        end
      end
    end
  end
end

function mean_pooling_forward{T}(input::Array{T}, output::Array{T}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  kernel_size = layer.kernel[1] * layer.kernel[2]

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*layer.stride[2] - layer.pad[2] + 1)
          wstart = max(1, (pw-1)*layer.stride[1] - layer.pad[1] + 1)
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)

          the_sum = 0.0
          for w = wstart:wend
            for h = hstart:hend
              @inbounds the_sum += input[w,h,c,n]
            end
          end
          @inbounds output[pw,ph,c,n] = the_sum / kernel_size
        end
      end
    end
  end
end

function max_pooling_backward{T}(input::Array{T}, output::Array{T}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  fill!(input, 0)
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          index = mask[pw,ph,c,n]
          idx_w = (index % width) + 1
          idx_h = div(index, width) + 1
          @inbounds input[idx_w, idx_h, c, n] += output[pw,ph,c,n]
        end
      end
    end
  end
end

function mean_pooling_backward{T}(input::Array{T}, output::Array{T}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  kernel_size = layer.kernel[1] * layer.kernel[2]

  fill!(input, 0)
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*layer.stride[2] - layer.pad[2] + 1)
          wstart = max(1, (pw-1)*layer.stride[1] - layer.pad[1] + 1)
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)

          @inbounds val = output[pw,ph,c,n] / kernel_size
          for w = wstart:wend
            for h = hstart:hend
              @inbounds input[w,h,c,n] += val
            end
          end
        end
      end
    end
  end
end

################################################################################
# Pooling in channels
################################################################################
function max_channel_pooling_forward{T}(input::Array{T}, output::Array{T}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)

  for n = 1:num
    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*layer.stride - layer.pad[1] + 1)
      cend   = min(cstart + layer.kernel - 1, channels)
      for w = 1:width
        for h = 1:height
          @inbounds output[w,h,pc,n] = input[w,h,cstart,n]
          @inbounds mask[w,h,pc,n] = cstart
        end
      end

      for c = cstart+1:cend
        for w = 1:width
          for h = 1:height
            @inbounds maxval = output[w,h,pc,n]
            @inbounds val = input[w,h,c,n]
            if val > maxval
              @inbounds output[w,h,pc,n] = val
              @inbounds mask[w,h,pc,n] = c
            end
          end
        end
      end
    end
  end
end

function mean_channel_pooling_forward{T}(input::Array{T}, output::Array{T}, integral::Array{T}, layer)
  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)
  one = convert(T, 1)
  neg_one = convert(T, -1)
  scale = 1/convert(T, layer.kernel)

  spatial_dim_T = width*height
  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = convert(Ptr{T}, input) + fea_dim*(n-1)
    output_ptr = convert(Ptr{T}, output) + output_fea_dim*(n-1)
    integral_ptr = convert(Ptr{T}, integral)

    # compute integral image
    BLAS.blascopy!(spatial_dim_T, input_ptr, 1, integral_ptr, 1)
    for c = 2:channels
      BLAS.blascopy!(spatial_dim_T, input_ptr + (c-1)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
      BLAS.axpy!(spatial_dim_T, one, integral_ptr + (c-2)*spatial_dim, 1,
          integral_ptr + (c-1)*spatial_dim, 1)
    end

    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*layer.stride - layer.pad[1] + 1)
      cend   = min(cstart + layer.kernel - 1, channels)
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

function max_channel_pooling_backward{T}(input::Array{T}, output::Array{T}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)

  fill!(input, 0)
  for n = 1:num
    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*layer.stride - layer.pad[1] + 1)
      cend   = min(cstart + layer.kernel - 1, channels)

      for w = 1:width
        for h = 1:height
          @inbounds input[w,h,mask[w,h,pc,n],n] += output[w,h,pc,n]
        end
      end
    end
  end
end

function mean_channel_pooling_backward{T}(input::Array{T}, output::Array{T}, layer)
  width, height, channels, num = size(input)
  pooled_chann = size(output, 3)
  scale = 1/convert(T, layer.kernel)

  fill!(input, 0)

  spatial_dim_T = width*height
  spatial_dim = spatial_dim_T * sizeof(T)
  fea_dim = spatial_dim * channels
  output_fea_dim = spatial_dim * pooled_chann

  for n = 1:num
    input_ptr = convert(Ptr{T}, input) + fea_dim*(n-1)
    output_ptr = convert(Ptr{T}, output) + output_fea_dim*(n-1)

    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*layer.stride - layer.pad[1] + 1)
      cend   = min(cstart + layer.kernel - 1, channels)
      output_ptr_pc = output_ptr + (pc-1)*spatial_dim

      for c = cstart:cend
        BLAS.axpy!(spatial_dim_T, scale, output_ptr_pc, 1,
            input_ptr + (c-1)*spatial_dim, 1)
      end
    end
  end
end
