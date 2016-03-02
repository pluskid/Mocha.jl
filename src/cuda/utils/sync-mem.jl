export SyncMem
export alloc_host_array, free_host_array, destory

import Base.mean

function alloc_host_array{T}(blob::CuTensorBlob{T})
  p = CudaRT.alloc_host(T, length(blob))
  p = convert(Ptr{T}, p)
  return pointer_to_array(p, size(blob))
end

function alloc_host_array{N}(T::Type, dims::NTuple{N, Int})
  p = CudaRT.alloc_host(T, prod(dims))
  p = convert(Ptr{T}, p)
  return pointer_to_array(p, dims)
end

function free_host_array(data::Array)
  CudaRT.free_host(convert(Ptr{Void}, pointer(data)))
end

type CuHostBlob{T, N} <: Blob{T, N}
  data      :: Array{Array{T, N}}         # one for each device
  cur_dev   :: CudaDevice
end
function CuHostBlob{T, N}(backend::GPUBackend, dtype::Type{T}, dims::NTuple{N, Int})
  data = Array(Array{T, N}, backend.dev_count)
  for i=1:backend.dev_count
    data[i] = alloc_host_array(dtype, dims)
  end
  cur_dev = backend.cur_dev
  return CuHostBlob(data, cur_dev)
end
CuHostBlob{T, N}(backend::GPUBackend, dev_blob::CuTensorBlob{T, N}) =
    CuHostBlob(backend, eltype(dev_blob), size(dev_blob))
get_data(blob::CuHostBlob) = @inbounds return blob.data[blob.cur_dev.ordinal + 1]
ndev(blob :: CuHostBlob) = length(blob.data)

function copy!{T}(dst :: CuHostBlob{T}, src :: CuTensorBlob{T})
  copy!(get_data(dst), src)
end
function copy_async!{T}(backend :: GPUBackend, dst :: CuTensorBlob{T}, src :: CuHostBlob{T})
  copy_async!(backend, dst, get_data(src))
end

function mean{T}(blob :: CuHostBlob{T})
  return mean(blob.data)
end

destroy(blob::CuHostBlob) = map(free_host_array, blob.data)

type SyncMem{T}
  dev_blob  :: CuTensorBlob{T}
  host_blob :: CuHostBlob{T}
end
function SyncMem{T}(backend::GPUBackend, blob :: CuTensorBlob{T})
    dev_blob = blob
    host_blob = CuHostBlob(backend, blob)
    return SyncMem{T}(dev_blob, host_blob)
end
function sync!(sync_mem :: SyncMem)
  copy!(sync_mem.host_blob, sync_mem.dev_blob)
end
function sync_all!(sync_mem :: SyncMem)
  @forall(sync_mem.host_blob,
    copy!(sync_mem.host_blob, sync_mem.dev_blob)
  )
end
function destroy(sync_mem :: SyncMem)
  destroy(sync_mem.dev_blob)
  destroy(sync_mem.host_blob)
end