using Benchmark

require("parallel-pool-module")

input = rand(28, 28, 20, 128)
kernel_dim = (5, 5)
stride_dim = (2, 2)

output = pool_single_proc(input, :max, kernel_dim, stride_dim)
output2 = pool_multiple_proc(input, :max, kernel_dim, stride_dim)
@assert all(abs(output-output2) .< 1e-10)

pool_single_proc() = pool_single_proc(input, :max, kernel_dim, stride_dim)
pool_multiple_proc() = pool_multiple_proc(input, :max, kernel_dim, stride_dim)

df = compare([pool_single_proc, pool_multiple_proc], 10)
