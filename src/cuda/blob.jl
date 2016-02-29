using .CUDA
using .CudaRT

export CuBlobDescriptor, CuPODBlobDescriptor, CuTensorBlobDescriptor, CuFilterBlobDescriptor
export CuTensorBlob
export get_ptr

type CuTensorBlob{T,N} <: Blob{T,N}
  ptrs      :: Array{CudaPtr} # one CudaPtr for each device
  shape     :: NTuple{N, Int}
  len       :: Int
  own_data  :: Bool
end
function CuTensorBlob{T, N}(dtype::Type{T}, dims::NTuple{N,Int})
  len = prod(dims)
  ptrs = Array(CudaPtr, CudaRT.get_dev_count())
  orig_dev = CudaRT.get_device()
  for i=1:CudaRT.get_dev_count()
    CudaRT.set_device(i - 1)
    @inbounds ptrs[i] = CudaRT.malloc(dtype, len)
  end
  CudaRT.set_device(orig_dev)
  return CuTensorBlob{T,N}(ptrs, dims, len, true)
end
get_ptr(blob::CuTensorBlob) = @inbounds return blob.ptrs[CudaRT.get_device() + 1]

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape

function copy!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CuBLAS.set_vector(src, get_ptr(dst))
end
function copy!{T}(dst :: Array{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CuBLAS.get_vector(get_ptr(src), dst)
end
function copy!{T}(dst :: CuTensorBlob{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CudaRT.copy!(get_ptr(dst), get_ptr(src), sizeof(dst))
end
function copy_async!{T}(backend::GPUBackend, dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CuBLAS.set_vector(src, get_ptr(dst), stream=backend.stream)
end
function fill!{T}(dst :: CuTensorBlob{T}, val)
  val_vec = Array(T, length(dst))
  fill!(val_vec, val)
  copy!(dst, val_vec)
end
function erase!{T}(dst :: CuTensorBlob{T})
  CudaRT.memset!(get_ptr(dst), 0, sizeof(dst))
end

function make_blob{N}(backend::GPUBackend, data_type::Type, dims::NTuple{N,Int})
  return CuTensorBlob(data_type, dims)
end
function replace_ptr(backend::GPUBackend, dst::CuTensorBlob, src::CuTensorBlob, offset::Int)
  @assert(length(dst.ptrs) == length(src.ptrs))
  @assert(length(dst.ptrs) == CudaRT.get_dev_count())
  destroy(dst)
  for i=1:CudaRT.get_dev_count()
    @inbounds dst.ptrs[i].p = src.ptrs[i].p + offset
  end
  dst.own_data = false
end
function reshape_blob{T,N}(backend::GPUBackend, blob::CuTensorBlob{T}, dims::NTuple{N,Int})
  @assert prod(dims) == length(blob)
  return CuTensorBlob{T,N}(blob.ptrs, dims, length(blob), blob.own_data)
end
function destroy(blob :: CuTensorBlob)
  if blob.own_data
    map(ptr -> ptr.p != 0 && CudaRT.free(ptr), blob.ptrs)
    map(ptr -> ptr.p = 0, blob.ptrs)
  end
end

