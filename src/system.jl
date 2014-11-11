export System
export init, shutdown

type System{T <: Backend}
  backend :: T
  layer_registry :: Dict{Layer, LayerState}
end
System{T<:Backend}(backend :: T) = System(backend, Dict{Layer, LayerState}())

function init(sys::System)
  init(sys.backend)
end
function shutdown(sys::System)
  shutdown(sys.backend)
end
