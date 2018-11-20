export Blob
export CPUBlob, NullBlob

import Base: eltype, size, length, ndims, copy!, fill!, show, randn!
export       eltype, size, length, ndims, copy!, fill!, erase!, show
export get_num, get_height, get_width, get_fea_size, to_array
export make_blob, make_zero_blob, reshape_blob

############################################################
# A blob is an abstract concept that is suppose
# to hold a 4-D tensor of data. The data could
# either live in CPU memory or GPU memory or
# whatever the backend is used to store the data.
############################################################
@compat abstract type Blob{T, N} end

############################################################
# The following should be implemented for a
# concrete Blob type. Note the following
# procedures are only provided for convenience
# and mainly for components that do not need
# to know the underlying backend (e.g. Filler).
############################################################
function eltype(blob :: Blob{T}) where {T}
  T
end

function ndims(blob :: Blob{T,N}) where {T,N}
  N
end
function size(blob :: Blob) # should return the size of data
  error("size not implemented for type $(typeof(blob))")
end
function destroy(blob :: Blob) # should destroy the blob
  error("destroy not implemented for type $(typeof(blob))")
end
function size(blob :: Blob{T,N}, dim :: Int) where {T,N}
  if dim < 0
    dim = N+1 + dim
  end

  return dim > N ? 1 : size(blob)[dim]
end
function length(blob :: Blob)
  return prod(size(blob))
end

function get_height(blob :: Blob)
  size(blob, 2)
end
function get_width(blob :: Blob)
  size(blob, 1)
end

function get_num(blob :: Blob)
  size(blob, -1)
end
function get_fea_size(blob :: Blob)
  prod(size(blob)[1:end-1])
end

function show(io::IO, blob :: Blob)
  shape = join(map(x -> "$x", size(blob)), " x ")
  print(io, "Blob($shape)")
end

function to_array(blob::Blob)
  array = Array{eltype(blob)}(size(blob))
  copy!(array, blob)
  array
end

function copy!(dst :: Array, src :: Blob) # should copy content of src to dst
  error("copy! not implemented from src type $(typeof(src)) to dest type $(typeof(dst))")
end
function copy!(dst :: Blob, src :: Array) # should copy content of src to dst
  error("copy! not implemented from src type $(typeof(src)) to dest type $(typeof(dst))")
end
function fill!(dst :: Blob, val) # should fill dst with val
  error("fill! not implemented for src value type $(typeof(val)) to dest type $(typeof(dst))")
end
function erase!(dst :: Blob)
  fill!(dst, 0)
end
function randn!(dst :: Blob) # should fill dst with iid standard normal variates
  error("randn! not implemented for type $(typeof(dst))")
end

############################################################
# A Dummy Blob type holding nothing
############################################################
struct NullBlob <: Blob{Nothing, 0}
end
function fill!(dst :: NullBlob, val)
  # do nothing
end
function show(io::IO, blob::NullBlob)
  print(io, "Blob()")
end

function destroy(blob::NullBlob)
  # do nothing
end
function make_blob(backend::Backend, data_type::Type, dims::Int...)
  make_blob(backend, data_type, dims)
end
function make_blob(backend::Backend, data::Array)
  blob = make_blob(backend, eltype(data), size(data))
  copy!(blob, data)
  return blob
end
function make_zero_blob(backend::Backend, data_type::Type, dims::NTuple{N,Int}) where {N}
  blob = make_blob(backend, data_type, dims)
  erase!(blob)
  return blob
end
function make_zero_blob(backend::Backend, data_type::Type, dims::Int...)
  make_zero_blob(backend, data_type, dims)
end

function reshape_blob(backend::Backend, blob::Blob, dims::Int...)
  reshape_blob(backend, blob, dims)
end

############################################################
# A Blob for CPU Computation
############################################################
struct CPUBlob{T <: AbstractFloat, N} <: Blob{T, N}
  data :: AbstractArray{T, N}
end
CPUBlob(t :: Type, dims::NTuple{N,Int}) where {N} = CPUBlob(Array{t}(dims)) 

function make_blob(backend::CPUBackend, data_type::Type, dims::NTuple{N,Int}) where {N}
  return CPUBlob(data_type, dims)
end

function reshape_blob(backend::CPUBackend, blob::CPUBlob{T,N1}, dims::NTuple{N2,Int}) where {T,N1,N2}
  @assert prod(dims) == length(blob)
  return CPUBlob{T,N2}(reshape(blob.data, dims))
end
function destroy(blob::CPUBlob)
  # do nothing... or is there anything that I could do?
end

size(blob::CPUBlob) = size(blob.data)

function copy!(dst :: Array{T}, src :: CPUBlob{T}) where {T}
  @assert length(dst) == length(src)
  dst[:] = src.data[:]
end
function copy!(dst :: CPUBlob{T}, src :: Array{T}) where {T}
  @assert length(dst) == length(src)
  dst.data[:] = src[:]
end
function copy!(dst :: CPUBlob{T}, src :: CPUBlob{T}) where {T}
  dst.data[:] = src.data[:]
end
function fill!(dst :: CPUBlob{T}, src) where {T}
  fill!(dst.data, src)
end
function randn!(dst :: CPUBlob{T}) where {T}
  randn!(dst.data)
end

function randn!(a::Array{Float32})
    # TODO This is hideously inefficient - check status of Julia issue
    # https://github.com/JuliaLang/julia/issues/9836
    @compat a[:] = map(Float32, randn(size(a)))
end
