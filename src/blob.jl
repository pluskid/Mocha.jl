export Blob
export CPUBlob, NullBlob

import Base: eltype, size, length
export       eltype, size, length, copy!, fill!

############################################################
# A blob is an abstract concept that is suppose
# to hold a 4-D tensor of data. The data could
# either live in CPU memory or GPU memory or
# whatever the backend is used to store the data.
############################################################
abstract Blob

############################################################
# The following should be implemented for a
# concrete Blob type. Note the following
# procedures are only provided for convenience
# and mainly for components that do not need
# to know the underlying backend (e.g. Filler).
############################################################
function eltype(blob :: Blob)
  error("Not implemented (should return the element type)")
end

function size(blob :: Blob)
  error("Not implemented (should return the size of data)")
end
function size(blob :: Blob, dim :: Int)
  size(blob)[dim]
end
function length(blob :: Blob)
  return prod(size(blob))
end

function copy!(dst :: Array, src :: Blob)
  error("Not implemented (should copy content of src to dst)")
end
function copy!(dst :: Blob, src :: Array)
  error("Not implemented (should copy content of src to dst)")
end
function fill!(dst :: Blob, val)
  error("Not implemented (should fill dst with val)")
end

############################################################
# A Dummy Blob type holding nothing
############################################################
type NullBlob <: Blob
end

############################################################
# A Blob for CPU Computation
############################################################
type CPUBlob{T <: FloatingPoint} <: Blob
  data :: Array{T}
end

eltype{T}(::CPUBlob{T}) = T
size(blob::CPUBlob) = size(blob.data)

function copy!{T}(dst :: Array{T}, src :: CPUBlob{T})
  @assert length(dst) == length(src)
  dst[:] = src.data[:]
end
function copy!{T}(dst :: CPUBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  dst.data[:] = src[:]
end

function fill!{T}(dst :: CPUBlob{T}, src)
  dst.data[:] = convert(eltype(dst.data), src)
end
