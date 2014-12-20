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
  if ndims(param) == 1 # a bias blob
    apply_l2_cons!(backend, param, cons.coefficient, length(param), 1)
  else # general weights
    apply_l2_cons!(backend, param, cons.coefficient, get_fea_size(param), get_num(param))
  end
end
