export System
export init, shutdown

type System{T <: Backend}
  backend :: T
end

function init(sys::System)
  init(sys.backend)
end
function shutdown(sys::System)
  shutdown(sys.backend)
end
