using Benchmark

#------------------------------------------------------------
# On my laptop
#
# julia:
# | Row | Function             | Average   | Relative | Replications |
# |-----|----------------------|-----------|----------|--------------|
# | 1   | "pool_single_proc"   | 0.0320213 | 1.0      | 50           |
# | 2   | "pool_multiple_proc" | 0.067476  | 2.10722  | 50           |
#
# julia -p 2:
# | Row | Function             | Average   | Relative | Replications |
# |-----|----------------------|-----------|----------|--------------|
# | 1   | "pool_single_proc"   | 0.0333833 | 1.0      | 50           |
# | 2   | "pool_multiple_proc" | 4.48791   | 134.436  | 50           |
#
#------------------------------------------------------------
# On a server with
# julia -p 8:
# | Row | Function             | Average   | Relative | Replications |
# |-----|----------------------|-----------|----------|--------------|
# | 1   | "pool_single_proc"   | 0.0387865 | 1.0      | 50           |
# | 2   | "pool_multiple_proc" | 5.88758   | 151.795  | 50           |

require("parallel-pool-module")

input = rand(28, 28, 20, 128)
kernel_dim = (5, 5)
stride_dim = (2, 2)

output = pool_single_proc(input, :max, kernel_dim, stride_dim)
output2 = pool_multiple_proc(input, :max, kernel_dim, stride_dim)
@assert all(abs.(output-output2) .< 1e-10)

pool_single_proc() = pool_single_proc(input, :max, kernel_dim, stride_dim)
pool_multiple_proc() = pool_multiple_proc(input, :max, kernel_dim, stride_dim)

df = compare([pool_single_proc, pool_multiple_proc], 50)
println("$df")
