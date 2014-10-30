export Regularizer
export NoRegu, L2Regu
export forward, backward

abstract Regularizer

type NoRegu <: Regularizer
  coefficient :: FloatingPoint # not used, just for consistent API
end
NoRegu() = NoRegu(0.0)

type L2Regu <: Regularizer
  coefficient :: FloatingPoint
end

############################################################
# No regularization
############################################################
# This function should return a number (the regularization) to be added to the objective function value
function forward(sys::System, regu :: NoRegu, param :: Blob)
  # 0, since no regularization
  return convert(eltype(param), 0)
end

# This function should compute the gradient of the regularizer and add it to the gradient blob. Note
# the gradient blob already contains computed gradient, make sure to ADD to instead of to overwrite it.
function backward(sys::System, regu :: NoRegu, param :: Blob, gradient :: Blob)
  # do nothing, since no regularization
end

############################################################
# L2 regularization
############################################################
function forward(sys::System{CPUBackend}, regu :: L2Regu, param :: Blob)
  return regu.coefficient * vecnorm(param.data)
end

function backward(sys::System{CPUBackend}, regu :: L2Regu, param:: Blob, gradient :: Blob)
  gradient.data += regu.coefficient * param.data
end

