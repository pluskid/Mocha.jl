export Backend, CPU, GPU
export System

abstract Backend

type CPU <: Backend; end
type CUDNN <: Backend; end

type System{T <: Backend}
  backend :: T
end
