export Backend, CPUBackend, AbstractCuDNNBackend
export init, shutdown, reset_registry

abstract Backend
function init(backend::Backend)
end
function shutdown(backend::Backend)
end
function reset_registry(backend::Backend)
  backend.layer_registry = Dict{String, LayerState}()
end

type CPUBackend{N} <: Backend
  layer_registry :: Dict{String, LayerState}

  pids :: NTuple{N, Int}

  CPUBackend(pids::NTuple{N,Int}) = begin
    for pid in pids
      if !in(pid, procs())
        error("$pid is not a valid process id")
      end
    end
    new(Dict{String, LayerState}(), pids)
  end
end

CPUBackend() = CPUBackend{1}((myid(),))
CPUBackend(pids::Vector{Int}) = CPUBackend{length(pids)}(tuple(pids...))

# This is forward declaration to allow some code to compile
# (especially testing codes) even if CUDA module is completely
# disabled. See test/layers/pooling.jl for example.
abstract AbstractCuDNNBackend <: Backend
