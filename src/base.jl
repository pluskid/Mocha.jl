export NumericRoot

export Backend, CPU, GPU
export System

# Numerical computations are only supported
# for number types that are subtype of this
# NumericRoot in Mocha
typealias NumericRoot FloatingPoint 

abstract Backend

type CPU <: Backend; end
type CUDNN <: Backend; end

type System{T <: Backend}
  backend :: T
end
