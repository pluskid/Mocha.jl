export Constraint, NoCons, L2Cons
export constrain!

abstract Constraint

immutable NoCons <: Constraint
  coefficient :: FloatingPoint # not used, just for consistent API
  every_n_iter :: Int          # also not used
end
NoCons() = NoCons(0.0, 0)

immutable L2Cons <: Constraint
  coefficient :: FloatingPoint
  every_n_iter :: Int
end
L2Cons(coefficient) = L2Cons(coefficient, 1)

############################################################
# No constraint
############################################################
function constrain!(backend::Backend, cons::NoCons, param)
  # do nothing if no constraints apply
end

############################################################
# L2 norm constraint on the weights
############################################################

function apply_l2_cons!{T <: FloatingPoint}(backend::CPUBackend, blob::CPUBlob{T},
                                            coef::FloatingPoint, ninputs::Int, nunits::Int)
  param = reshape(blob.data, (ninputs, nunits))
  # we constrain each column vector
  for i = 1:nunits
    # compute norm and scale using blas
    norm = vecnorm(param[:, i])
    if norm > coef
      scale_factor =  (1. / norm) * coef
      offset = sizeof(T) * (i-1) * ninputs
      BLAS.scal!(ninputs, convert(T, scale_factor), convert(Ptr{T}, param) + offset, 1)
    end
  end
end

# this constraints a given blob along the last dimension that is not of size 1
# it is a bit ugly but was the easiest way to implement it for now
function constrain!(backend :: Backend, cons :: L2Cons, param :: Blob)
  W = size(param, 1)   # source dim in fully connected
  H = size(param, 2)   # target dim in fully connected
  C = size(param, 3)
  N = size(param, 4)   # number of filters in convolution
  if H == 1 && N == 1 && C == 1
    # only width left ... this is a bias ... lets constrain that
    apply_l2_cons!(backend, param, cons.coefficient, W, 1)
  elseif N == 1 && C == 1
    # we have only one channel and num -> constrain target dim
    apply_l2_cons!(backend, param, cons.coefficient, W, H)
  else
    # constrain on N -> e.g. the number if units for convolutional filters
    apply_l2_cons!(backend, param, cons.coefficient, W*H*C, N)
  end
end
