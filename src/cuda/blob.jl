using .CUDA

export CuBlobDescriptor, CuPODBlobDescriptor, CuTensorBlobDescriptor, CuFilterBlobDescriptor
export CuTensorBlob

immutable CuTensorBlob{T<:FloatingPoint} <: Blob
  ptr   :: CuPtr
  shape :: NTuple{4, Int}
  len   :: Int
end
function CuTensorBlob{T<:FloatingPoint}(dtype::Type{T}, w::Int, h::Int, c::Int, n::Int)
  len = n*c*h*w
  ptr = CUDA.cualloc(dtype, len)
  return CuTensorBlob{T}(ptr, (w,h,c,n), len)
end

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape
eltype{T}(b::CuTensorBlob{T}) = T

function copy!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CuBLAS.set_vector(src, dst.ptr)
end
function copy!{T}(dst :: Array{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CuBLAS.get_vector(src.ptr, dst)
end
function copy!{T}(dst :: CuTensorBlob{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  @CUDA.cucall(:cuMemcpy, (Ptr{Void}, Ptr{Void}, Cint), dst.ptr.p, src.ptr.p, length(dst)*sizeof(T))
end
function fill!{T}(dst :: CuTensorBlob{T}, val)
  val_vec = Array(T, length(dst))
  fill!(val_vec, val)
  copy!(dst, val_vec)
end
function erase!{T}(dst :: CuTensorBlob{T})
  @CUDA.cucall(:cuMemsetD8_v2, (Ptr{Void}, Cuchar, Csize_t), dst.ptr.p, 0, length(dst)*sizeof(T))
end

function cudnn_make_tensor_blob(dtype::Type, dims::NTuple{4,Int})
  return CuTensorBlob(dtype, dims...)
end

function make_blob(backend::GPUBackend, data_type::Type, dims::NTuple{4,Int})
  return cudnn_make_tensor_blob(data_type, dims)
end
function reshape_blob{T}(backend::GPUBackend, blob::CuTensorBlob{T}, dims::NTuple{4,Int})
  @assert prod(dims) == length(blob)
  return CuTensorBlob{T}(blob.ptr, dims, length(blob))
end
function destroy(blob :: CuTensorBlob)
  if blob.ptr.p != 0
    CUDA.free(blob.ptr)
    blob.ptr.p = 0
  end
end

