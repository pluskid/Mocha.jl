function pool_single_proc(input, pool, kernel, stride)
  width, height, channels, num = size(input)
  pooled_width  = int(ceil(float(width -kernel[1]) / stride[1]))+1
  pooled_height = int(ceil(float(height-kernel[2]) / stride[2]))+1
  kernel_size = kernel[1] * kernel[2]

  output = zeros(pooled_width, pooled_height, channels, num)

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*stride[2] + 1)
          wstart = max(1, (pw-1)*stride[1] + 1)
          hend = min(hstart + kernel[2] - 1, height)
          wend = min(wstart + kernel[1] - 1, width)
          if pool == :max
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
          elseif pool == :mean
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

  return output
end

function pool_multiple_proc(input, pool, kernel, stride)
  num = size(input, 4)
  outputs = [@spawn pool_multiple_proc_impl(input[:,:,:,n], pool, kernel, stride) for n = 1:num]
  return cat(4, map(fetch, outputs)...)
end
function pool_multiple_proc_impl(input, pool, kernel, stride)
  width, height, channels = size(input)
  pooled_width  = int(ceil(float(width -kernel[1]) / stride[1]))+1
  pooled_height = int(ceil(float(height-kernel[2]) / stride[2]))+1
  kernel_size = kernel[1] * kernel[2]

  output = zeros(pooled_width, pooled_height, channels)

  for c = 1:channels
    for ph = 1:pooled_height
      for pw = 1:pooled_width
        hstart = max(1, (ph-1)*stride[2] + 1)
        wstart = max(1, (pw-1)*stride[1] + 1)
        hend = min(hstart + kernel[2] - 1, height)
        wend = min(wstart + kernel[1] - 1, width)
        if pool == :max
          maxval = -Inf
          maxw = 0
          maxh = 0
          for w = wstart:wend
            for h = hstart:hend
              @inbounds val = input[w,h,c]
              if val > maxval
                maxval = val
                maxw = w
                maxh = h
              end
            end
          end
          @inbounds output[pw,ph,c] = maxval
        elseif pool == :mean
          the_sum = 0.0
          for w = wstart:wend
            for h = hstart:hend
              @inbounds the_sum += input[w,h,c]
            end
          end
          @inbounds output[pw,ph,c] = the_sum / kernel_size
        end
      end
    end
  end

  return output
end
