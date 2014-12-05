export Backend, CPUBackend, AbstractCuDNNBackend
export init, shutdown, registry_reset, registry_put

abstract Backend
typealias ParameterRegistry Dict{String, Vector{AbstractParameter}}

function init(backend::Backend)
end
function shutdown(backend::Backend)
end
function registry_reset(backend::Backend)
  backend.param_registry = ParameterRegistry()
end
function registry_put(backend::Backend, key::String, params::Vector)
  # convert Vector{Parameter} to Vector{AbstractParameter}
  backend.param_registry[key] = convert(Vector{AbstractParameter}, params)
end
function registry_get(backend::Backend, key::String)
  return get(backend.param_registry, key, nothing)
end

type CPUBackend{N} <: Backend
  param_registry :: ParameterRegistry

  pids :: Vector{Int}

  CPUBackend(pids::Vector{Int}) = begin
    for pid in pids
      if !in(pid, procs())
        error("$pid is not a valid process id")
      end
    end
    pids = copy(pids)
    if !in(myid(), pids)
      push!(pids, myid())
    end
    new(ParameterRegistry(), pids)
  end
end

CPUBackend() = CPUBackend{1}([myid()])
CPUBackend(pids::Vector{Int}) = CPUBackend{length(pids)}(pids)

# This is forward declaration to allow some code to compile
# (especially testing codes) even if CUDA module is completely
# disabled. See test/layers/pooling.jl for example.
abstract AbstractGPUBackend <: Backend
