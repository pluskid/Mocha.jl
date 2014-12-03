export System
export init, reset, shutdown

type System{T <: Backend}
  backend :: T
  layer_registry :: Dict{String, LayerState}
end
System{T<:Backend}(backend :: T) = System(backend, Dict{String, LayerState}())

function init(sys::System)
  init(sys.backend)
end
function reset(sys::System)
  sys.layer_registry = Dict{String, LayerState}()
end
function shutdown(sys::System)
  shutdown(sys.backend)
end
