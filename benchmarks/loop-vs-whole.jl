using Benchmark
################################################################################
# Benchmark to see whether loop is faster than matrix operation
#
# | Row | Function     | Average   | Relative | Replications |
# |-----|--------------|-----------|----------|--------------|
# | 1   | "relu_loop"  | 0.0283602 | 1.0      | 30           |
# | 2   | "relu_whole" | 0.0611887 | 2.15755  | 30           |
#
# | Row | Function         | Average   | Relative | Replications |
# |-----|------------------|-----------|----------|--------------|
# | 1   | "relu_bwd_loop"  | 0.0314793 | 1.0      | 30           |
# | 2   | "relu_bwd_whole" | 0.0696191 | 2.21158  | 30           |
################################################################################

function relu_loop(input)
  input = copy(input)
  for i = 1:length(input)
    input[i] = max(input[i], 0)
  end
  return input
end

function relu_whole(input)
  input = copy(input)
  input = max(input, 0)
  return input
end

function relu_bwd_loop(input, grad)
  grad = copy(grad)
  for i = 1:length(input)
    if input[i] <= 0
      grad[i] = 0
    end
  end
  return grad
end
function relu_bwd_whole(input, grad)
  grad = copy(grad)
  grad .*= (input .> 0)
  return grad
end

input = rand(28, 28, 50, 128)
i1 = relu_loop(input)
i2 = relu_whole(input)
@assert all(abs(i1-i2) .< 1e-10)

grad = rand(size(input))

relu_loop() = relu_loop(input)
relu_whole() = relu_whole(input)
relu_bwd_loop() = relu_bwd_loop(i1, grad)
relu_bwd_whole() = relu_bwd_whole(i1, grad)

df = compare([relu_loop, relu_whole], 30)
println("$df")

df = compare([relu_bwd_loop, relu_bwd_whole], 30)
println("$df")
