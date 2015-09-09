if haskey(ENV, "MOCHA_USE_CUDA")
  const test_gpu = true
else
  const test_gpu = false
end
const test_cpu   = true

using Mocha
using Base.Test

if test_cpu
  backend_cpu = CPUBackend()
  init(backend_cpu)
end

if test_gpu
  backend_gpu = GPUBackend()
  init(backend_gpu)
end

# run test in the whole directory, latest modified files
# are run first, this makes waiting time shorter when writing
# or modifying unit-tests
function test_dir(dir)
  map(reverse(Mocha.glob(dir, r".*\.jl$", sort_by=:mtime))) do file
    include("$dir/$file")
  end
end

############################################################
# Solvers
############################################################
include("solvers/test-adam-solver.jl")
include("solvers/test-sgd-solver.jl")

############################################################
# Network
############################################################
include("net/topology.jl")
include("net/test-gradient-simple.jl")

############################################################
# Utilities functions
############################################################
include("utils/ref-count.jl")
include("utils/glob.jl")
include("utils/blas.jl")
include("utils/blob-reshape.jl")

if test_gpu
  include("cuda/padded-copy.jl")
  include("cuda/cuvec.jl")
  include("cuda/mocha.jl")
  include("cuda/cublas.jl")
end

############################################################
# Activation Functions
############################################################
include("neurons/relu.jl")
include("neurons/sigmoid.jl")
include("neurons/tanh.jl")

############################################################
# Regularizers
############################################################
include("regularizers/l2.jl")
include("regularizers/l1.jl")

############################################################
# Constraints
############################################################
include("constraints/l2.jl")

############################################################
# Data Transformers
############################################################
include("data-transformers.jl")


############################################################
# Layers
############################################################
test_dir("layers")

if test_gpu
  shutdown(backend_gpu)
end
if test_cpu
  shutdown(backend_cpu)
end
