import LinearAlgebra

export Constraint, NoCons, L2Cons
export constrain!

@compat abstract type Constraint end

struct NoCons <: Constraint
  threshold  :: AbstractFloat  # not used, just for consistent API
  every_n_iter :: Int          # also not used
end
NoCons() = NoCons(0.0, 0)

struct L2Cons <: Constraint
  threshold    :: AbstractFloat
  every_n_iter :: Int
end
L2Cons(threshold) = L2Cons(threshold, 1)

############################################################
# No constraint
############################################################
function constrain!(backend::Backend, cons::NoCons, param)
  # do nothing if no constraints apply
end

############################################################
# L2 norm constraint on the weights
############################################################

function apply_l2_cons!(backend::CPUBackend, blob::CPUBlob{T},
                                            threshold::AbstractFloat, ninputs::Int, nunits::Int) where {T <: AbstractFloat}
  param = reshape(blob.data, (ninputs, nunits))
  # we constrain each column vector
  for i = 1:nunits
    # compute norm and scale using blas
    norm = LinearAlgebra.norm(param[:, i])
    if norm > threshold
      scale_factor =  (1. / norm) * threshold
      offset = sizeof(T) * (i-1) * ninputs
      LinearAlgebra.BLAS.scal!(ninputs, convert(T, scale_factor), pointer(param) + offset, 1)
    end
  end
end

# this constraints a given blob along the last dimension that is not of size 1
# it is a bit ugly but was the easiest way to implement it for now
function constrain!(backend :: Backend, cons :: L2Cons, param :: Blob)
  if ndims(param) == 1 # a bias blob
    apply_l2_cons!(backend, param, cons.threshold, length(param), 1)
  else # general weights
    apply_l2_cons!(backend, param, cons.threshold, get_fea_size(param), get_num(param))
  end
end
