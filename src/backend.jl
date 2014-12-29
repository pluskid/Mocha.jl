export Backend, CPUBackend, AbstractGPUBackend
export init, shutdown, registry_reset, registry_put

abstract Backend
typealias ParameterRegistry Dict{String, Vector{AbstractParameter}}

import Base.show
export show
show(io::IO, backend::Backend) = show(io, typeof(backend))

function init(backend::Backend)
end
function shutdown(backend::Backend)
  registry_reset(backend)
end
function registry_reset(backend::Backend)
  map(params -> map(destroy, params), backend.param_registry)
  backend.param_registry = ParameterRegistry()
end
function registry_put(backend::Backend, key::String, params::Vector)
  if haskey(backend.param_registry, key)
    map(destroy, backend.param_registry[key])
  end

  # we keep a reference to the parameters, so that even those the original
  # network is destroyed, we still have valid access to those trained parameters
  backend.param_registry[key] = AbstractParameter[share_parameter(backend, p) for p in params]
end
function registry_get(backend::Backend, key::String)
  return get(backend.param_registry, key, nothing)
end

type CPUBackend <: Backend
  param_registry :: ParameterRegistry

  CPUBackend() = new(ParameterRegistry())
end

# This is forward declaration to allow some code to compile
# (especially testing codes) even if CUDA module is completely
# disabled. See test/layers/pooling.jl for example.
abstract AbstractGPUBackend <: Backend
