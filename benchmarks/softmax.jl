using Benchmark

################################################################################
# Test how matrix operations performance compared with elementwise operations
# when @inbounds annotation is turned on.
#
# | Row | Function         | Average     | Relative | Replications |
# |-----|------------------|-------------|----------|--------------|
# | 1   | "softmax_matrix" | 0.00168359  | 2.63545  | 50           |
# | 2   | "softmax_elem"   | 0.000638823 | 1.0      | 50           |
################################################################################

function softmax_matrix(input, the_output)
  output = copy(input)
  output .-= maximum(output, 3) # subtract max along channel dimension
  output = exp(output)
  output ./= sum(output, 3) # normalize along channel dimension
  copy!(the_output, output)
end

function softmax_elem(input, output)
  width, height, channels, num = size(input)

  for w = 1:width
    for h = 1:height
      for n = 1:num
        maxval = -Inf
        @simd for c = 1:channels
          @inbounds maxval = max(maxval, input[w,h,c,n])
        end
        @simd for c = 1:channels
          @inbounds output[w,h,c,n] = exp(input[w,h,c,n]-maxval)
        end
        the_sum = 0.0
        @simd for c = 1:channels
          @inbounds the_sum += output[w,h,c,n]
        end
        @simd for c = 1:channels
          @inbounds output[w,h,c,n] /= the_sum
        end
      end
    end
  end
end

input = rand(5, 5, 10, 128)
o1 = rand(size(input))
o2 = rand(size(input))
softmax_matrix() = softmax_matrix(input, o1)
softmax_elem() = softmax_elem(input, o2)

softmax_matrix()
softmax_elem()
@assert all(abs.(o1-o2) .< 1e-10)

df = compare([softmax_matrix, softmax_elem], 50)
println("$df")
