export System
export init, shutdown

type System{T <: Backend}
  backend :: T

  regularization_coef :: FloatingPoint
  learning_rate :: FloatingPoint
  momentum :: FloatingPoint
  max_iter :: Int
end

function init(sys::System)
  init(sys.backend)
end
function shutdown(sys::System)
  shutdown(sys.backend)
end
