export Regularizer
export Regularization

module Regularization

abstract T

abstract Nothing <: T # No regularization
abstract L2 <: T      # L2 norm regularization

end # module Regularization

type Regularizer{T1 <: Regularization.T}
  regularization :: T1
  coefficient    :: NumericRoot
end
Regularizer(regu :: Regularization.T, coef :: Number) = Regularizer(regu, convert(NumericRoot, coef))
Regularizer(regu :: Regularization.Nothing) = Regularizer(regu, 0)

############################################################
# No regularization
############################################################
# This function should return a number (the regularization) to be added to the objective function value
function forward(sys::System, regu :: Regularizer{Regularization.Nothing}, param :: Blob)
  # 0, since no regularization
  return convert(eltype(param), 0)
end

# This function should compute the gradient of the regularizer and add it to the gradient blob. Note
# the gradient blob already contains computed gradient, make sure to ADD to instead of to overwrite it.
function backward(sys::System, regu :: Regularizer{Regularization.Nothing}, param :: Blob, gradient :: Blob)
  # do nothing, since no regularization
end

############################################################
# L2 regularization
############################################################
function forward(sys::System{CPU}, regu :: Regularizer{Regularization.L2}, param :: Blob)
  return regu.coefficient * vecnorm(param.data)
end

function backward(sys::System{CPU}, regu :: Regularizer{Regularization.L2}, param:: Blob, gradient :: Blob)
  gradient.data += regu.coefficient * param.data
end
