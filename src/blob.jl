export Blob
export CPUBlob, NullBlob

import Base:    eltype, size, getindex, setindex!, endof
export backend, eltype, size, getindex, setindex!, endof

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
function backend(blob :: Blob)
  error("Not implemented (should return the backend)")
end

function eltype(blob :: Blob)
  error("Not implemented (should return the element type)")
end

function size(blob :: Blob)
  error("Not implemented (should return the size of data)")
end
function endof(blob :: Blob)
  prod(size(blob))
end

function getindex(blob :: Blob, idx...)
  error("Not implemented (should return data at given idx)")
end

function setindex!(blob :: Blob, value, idx...)
  error("Not implemented (should set value at given idx)")
end


############################################################
# A Dummy Blob type holding nothing
############################################################
type NullBlob <: Blob
end

############################################################
# A Blob for CPU Computation
############################################################
type CPUBlob{T <: NumericRoot} <: Blob
  data :: Array{T}
end

backend(::CPUBlob) = CPU()
eltype{T}(::CPUBlob{T}) = T

size(blob::CPUBlob) = size(blob.data)
getindex(blob::CPUBlob,idx...) = getindex(blob.data,idx...)
setindex!(blob::CPUBlob,idx...) = setindex!(blob.data,idx...)
