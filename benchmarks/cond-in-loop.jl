using Benchmark

################################################################################
# We could decide which pooling to use inside a loop, or we could
# write a different loop for different pooling. The latter leads
# to code duplication, but better performance. This benchmark is
# to show how much performance different this two ways have. It
# turns out the difference is not so significant for this case.
#
# | Row | Function       | Average | Relative | Replications |
# |-----|----------------|---------|----------|--------------|
# | 1   | "loop_in_cond" | 5.44311 | 1.0      | 10           |
# | 2   | "cond_in_loop" | 5.88683 | 1.08152  | 10           |
################################################################################

function cond_in_loop(input, kernel, pooling)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)
  kernel_size = prod(kernel)

  for n = 1:num
    for c = 1:channels
      for pw = 1:pooled_width
        for ph = 1:pooled_height
          hstart = ph
          wstart = pw
          hend = min(hstart + kernel[2] - 1, height)
          wend = min(wstart + kernel[1] - 1, width)
          if pooling == :max
            maxval = -Inf
            for u = wstart:wend
              for v = hstart:hend
                maxval = max(maxval, input[u,v,c,n])
              end
            end
            output[pw,ph,c,n] = maxval
          elseif pooling == :mean
            the_sum = 0
            for u = wstart:wend
              for v = hstart:hend
                the_sum += input[u,v,c,n]
              end
            end
            output[pw,ph,c,n] = the_sum / kernel_size
          end
        end
      end
    end
  end

  return output
end

function loop_in_cond(input, kernel, pooling)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)
  if pooling == :max
    for n = 1:num
      for c = 1:channels
        for pw = 1:pooled_width
          for ph = 1:pooled_height
            hstart = ph
            wstart = pw
            hend = min(hstart + kernel[2] - 1, height)
            wend = min(wstart + kernel[1] - 1, width)
            maxval = -Inf
            for u = wstart:wend
              for v = hstart:hend
                maxval = max(maxval, input[u,v,c,n])
              end
            end
            output[pw,ph,c,n] = maxval
          end
        end
      end
    end
  elseif pooling == :mean
    kernel_size = prod(kernel)

    for n = 1:num
      for c = 1:channels
        for pw = 1:pooled_width
          for ph = 1:pooled_height
            hstart = ph
            wstart = pw
            hend = min(hstart + kernel[2] - 1, height)
            wend = min(wstart + kernel[1] - 1, width)
            the_sum = 0
            for u = wstart:wend
              for v = hstart:hend
                the_sum += input[u,v,c,n]
              end
            end
            output[pw,ph,c,n] = the_sum / kernel_size
          end
        end
      end
    end
  end

  return output
end

input = rand(28, 28, 50, 128)
kernel = (5,5)
cond_in_loop() = (cond_in_loop(input,kernel,:max),
                  cond_in_loop(input,kernel,:mean))
loop_in_cond() = (loop_in_cond(input,kernel,:max),
                  loop_in_cond(input,kernel,:mean))
maxpool1, meanpool1 = cond_in_loop()
maxpool2, meanpool2 = cond_in_loop()
@assert all(abs.(maxpool1-maxpool2) .< 1e-10)
@assert all(abs.(meanpool1-meanpool2) .< 1e-10)

println("Running benchmark")
df = compare([loop_in_cond, cond_in_loop], 10)
println("$df")
