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
          hstart = (ph-1)*layer.stride[2] - layer.pad[2] + 1
          wstart = (pw-1)*layer.stride[1] - layer.pad[1] + 1
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)
          hstart = max(hstart, 1)
          wstart = max(wstart, 1)

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
          hstart = (ph-1)*layer.stride[2] - layer.pad[2] + 1
          wstart = (pw-1)*layer.stride[1] - layer.pad[1] + 1
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)
          hstart = max(hstart, 1)
          wstart = max(wstart, 1)

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
          hstart = (ph-1)*layer.stride[2] - layer.pad[2] + 1
          wstart = (pw-1)*layer.stride[1] - layer.pad[1] + 1
          hend = min(hstart + layer.kernel[2] - 1, height)
          wend = min(wstart + layer.kernel[1] - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

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

