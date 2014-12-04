export Backend, CPUBackend, AbstractCuDNNBackend
export init, shutdown

abstract Backend
function init(backend::Backend)
end
function shutdown(backend::Backend)
end

type CPUBackend{N} <: Backend
  pids :: NTuple{N, Int}

  CPUBackend(pids::NTuple{N,Int}) = begin
    for pid in pids
      if !in(pid, procs())
        error("$pid is not a valid process id")
      end
    end
    new(pids)
  end
end

CPUBackend() = CPUBackend{1}((myid(),))
CPUBackend(pids::Vector{Int}) = CPUBackend{length(pids)}(tuple(pids...))

# This is forward declaration to allow some code to compile
# (especially testing codes) even if CUDA module is completely
# disabled. See test/layers/pooling.jl for example.
abstract AbstractCuDNNBackend <: Backend
