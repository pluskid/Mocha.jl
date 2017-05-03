using Benchmark

################################################################################
# A benchmark for determining ways of implementing pooling. It turns out creating
# subarray in julia is very slow.
#
# | Row | Function              | Average  | Relative | Replications |
# |-----|-----------------------|----------|----------|--------------|
# | 1   | "pool_with_subarray"  | 10.3109  | 35.1015  | 10           |
# | 2   | "pool_with_iter"      | 0.329981 | 1.12336  | 10           |
# | 3   | "pool_with_iter_cond" | 0.293746 | 1.0      | 10           |
# | 4   | "pool_with_getidx"    | 1.94563  | 6.62353  | 10           |
################################################################################

function pool_with_subarray(input, kernel)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)

  for n = 1:num
    for c = 1:channels
      for pw = 1:pooled_width
        for ph = 1:pooled_height
          hstart = ph
          wstart = pw
          hend = min(hstart + kernel[2] - 1, height)
          wend = min(wstart + kernel[1] - 1, width)
          region = sub(input, wstart:wend, hstart:hend, c, n)
          output[pw,ph,c,n] = maximum(region)
        end
      end
    end
  end

  return output
end

function pool_with_getidx(input, kernel)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)

  for n = 1:num
    for c = 1:channels
      for pw = 1:pooled_width
        for ph = 1:pooled_height
          hstart = ph
          wstart = pw
          hend = min(hstart + kernel[2] - 1, height)
          wend = min(wstart + kernel[1] - 1, width)
          output[pw,ph,c,n] = maximum(input[wstart:wend, hstart:hend, c, n])
        end
      end
    end
  end

  return output
end

function pool_with_iter(input, kernel)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)

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

  return output
end

function pool_with_iter_cond(input, kernel)
  width, height, channels, num = size(input)
  pooled_width = width - kernel[1] + 1
  pooled_height = height - kernel[2] + 1

  output = zeros(pooled_width, pooled_height, channels, num)

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
              val = input[u,v,c,n]
              if val > maxval
                maxval = val
              end
            end
          end
          output[pw,ph,c,n] = maxval
        end
      end
    end
  end

  return output
end

input = rand(28, 28, 50, 128)
kernel = (5,5)
pool_with_subarray() = pool_with_subarray(input, kernel)
pool_with_getidx() = pool_with_getidx(input, kernel)
pool_with_iter() = pool_with_iter(input, kernel)
pool_with_iter_cond() = pool_with_iter_cond(input, kernel)

@assert all(abs.(pool_with_subarray() - pool_with_iter()) .< 1e-10)
@assert all(abs.(pool_with_subarray() - pool_with_getidx()) .< 1e-10)
@assert all(abs.(pool_with_subarray() - pool_with_iter_cond()) .< 1e-10)

println("Running benchmark")
df = compare([pool_with_subarray, pool_with_iter, pool_with_iter_cond, pool_with_getidx], 10)
println("$df")

