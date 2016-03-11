#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
using .CUDA
using .CudaRT

export CuBlobDescriptor, CuPODBlobDescriptor, CuTensorBlobDescriptor, CuFilterBlobDescriptor
export CuTensorBlob
export get_ptr

type CuTensorBlob{T,N} <: Blob{T,N}
  ptrs      :: Array{CudaPtr}         # one CudaPtr for each device
  cur_dev   :: CudaDevice
  shape     :: NTuple{N, Int}
  len       :: Int
  own_data  :: Bool
end
function CuTensorBlob{T, N}(backend::GPUBackend, dtype::Type{T}, dims::NTuple{N,Int})
  len = prod(dims)
  ptrs = Array(CudaPtr, backend.dev_count)
  orig_dev = backend.cur_dev.ordinal
  for dev=1:backend.dev_count
    set_dev_id(backend, dev - 1)
    @inbounds ptrs[dev] = CudaRT.malloc(dtype, len)
  end
  set_dev_id(backend, orig_dev)
  return CuTensorBlob{T,N}(ptrs, backend.cur_dev, dims, len, true)
end
get_ptr(blob::CuTensorBlob) = @inbounds return blob.ptrs[blob.cur_dev.ordinal + 1]

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape
ndev(b::CuTensorBlob) = length(b.ptrs)

function copy!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CudaRT.copy!(get_ptr(dst), convert(Ptr{Void}, pointer(src)), sizeof(dst))
end
function copy_all!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  @forall(dst, 
    copy!(dst, src)
  )
end
function copy!{T}(dst :: Array{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CudaRT.copy!(convert(Ptr{Void}, pointer(dst)), get_ptr(src), sizeof(src))
end
function copy!{T}(dst :: CuTensorBlob{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CudaRT.copy!(get_ptr(dst), get_ptr(src), sizeof(dst))
end
function copy_async!{T}(backend::GPUBackend, dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CudaRT.copy_async!(get_ptr(dst), convert(Ptr{Void}, pointer(src)), sizeof(dst), get_stream(backend))
end
function fill!{T}(dst :: CuTensorBlob{T}, val)
  val_vec = Array(T, length(dst))
  fill!(val_vec, val)
  copy!(dst, val_vec)
end
function fill_all!{T}(dst :: CuTensorBlob{T}, val)
  val_vec = Array(T, length(dst))
  fill!(val_vec, val)
  @forall(dst, 
    copy!(dst, val_vec)
  )

end
function erase!{T}(dst :: CuTensorBlob{T})
  CudaRT.memset!(get_ptr(dst), 0, sizeof(dst))
end
function erase_all!{T}(dst :: CuTensorBlob{T})
  @forall(dst, 
    erase!(dst)
  )
end

function make_blob{N}(backend::GPUBackend, data_type::Type, dims::NTuple{N,Int})
  return CuTensorBlob(backend, data_type, dims)
end
function make_blob(backend::GPUBackend, data::Array)
  blob = make_blob(backend, eltype(data), size(data))
  copy_all!(blob, data)
  return blob
end
function make_zero_blob{N}(backend::GPUBackend, data_type::Type, dims::NTuple{N,Int})
  blob = make_blob(backend, data_type, dims)
  erase_all!(blob)
  return blob
end
function replace_ptr(backend::GPUBackend, dst::CuTensorBlob, src::CuTensorBlob, offset::Int)
  @assert(length(dst.ptrs) == length(src.ptrs))
  @assert(length(dst.ptrs) == backend.dev_count)
  destroy(dst)
  for i=1:backend.dev_count
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

# pairwise mean: devx = (devx + devy) * 0.5
function mean_async!{T}(backend::GPUBackend, blob::CuTensorBlob{T}, tmp::CuTensorBlob{T}, devx::Int, devy::Int)
  @inbounds set_dev(backend, devx)
  @inbounds CudaRT.copy_async!(tmp.ptrs[devx + 1], blob.ptrs[devy + 1], sizeof(blob), get_stream(backend))
  @inbounds CuVec.mean!(backend, eltype(blob), blob.ptrs[devx + 1].p, tmp.ptrs[devx + 1].p, length(blob))
end
function mean!{T}(backend::GPUBackend, blob::CuTensorBlob{T}, tmp::CuTensorBlob{T})
#TODO: Fix this to add any number of GPUs support.
  if backend.dev_count == 2
    # 1 -> 0
    @inbounds mean_async!(backend, blob, tmp, 0, 1)
    @inbounds CudaRT.sync_stream(backend.streams[1])
    # 0 -> 1
    @inbounds CudaRT.copy!(blob.ptrs[2], blob.ptrs[1], sizeof(blob))
  elseif backend.dev_count == 4
    # 1 -> 0
    @inbounds mean_async!(backend, blob, tmp, 0, 1)
    # 3 -> 2
    @inbounds mean_async!(backend, blob, tmp, 2, 3)
    @inbounds CudaRT.sync_stream(backend.streams[1])
    @inbounds CudaRT.sync_stream(backend.streams[3])
    # 2 -> 0
    @inbounds mean_async!(backend, blob, tmp, 0, 2)
    @inbounds CudaRT.sync_stream(backend.streams[1])
    # 0 -> 2
    @inbounds CudaRT.copy!(blob.ptrs[3], blob.ptrs[1], sizeof(blob))
    # 0 -> 1
    @inbounds CudaRT.copy_async!(blob.ptrs[2], blob.ptrs[1], sizeof(blob), backend.streams[1])
    # 2 -> 3
    @inbounds CudaRT.copy_async!(blob.ptrs[4], blob.ptrs[3], sizeof(blob), backend.streams[3])
    @inbounds CudaRT.sync_stream(backend.streams[1])
    @inbounds CudaRT.sync_stream(backend.streams[3])
  end
end
