if haskey(ENV, "MOCHA_USE_CUDA")
  const test_cuda = true
else
  const test_cuda = false
end
if haskey(ENV, "MOCHA_USE_OPENCL")
  const test_opencl = true
else
  const test_opencl = false
end
const test_cpu   = true

using Mocha
using Base.Test

if test_cpu
  backend_cpu = CPUBackend()
  init(backend_cpu)
end

if test_cuda
  backend_cuda = CUDABackend()
  init(backend_cuda)
end

if test_opencl
  backend_opencl = OpenCLBackend()
  init(backend_opencl)
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

if test_cuda
  include("cuda/padded-copy.jl")
  include("cuda/cuvec.jl")
  include("cuda/mocha.jl")
  include("cuda/cublas.jl")
end

if test_opencl
  warn("TODO: OpenCL utilities tests")
end

############################################################
# Activation Functions
############################################################
include("neurons/relu.jl")
include("neurons/sigmoid.jl")
include("neurons/tanh.jl")
include("neurons/exponential.jl")

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

if test_opencl
  shutdown(backend_opencl)
end
if test_cuda
  shutdown(backend_cuda)
end
if test_cpu
  shutdown(backend_cpu)
end
