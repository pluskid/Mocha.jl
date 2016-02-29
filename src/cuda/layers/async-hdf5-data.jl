function alloc_host_array{T}(backend::GPUBackend, blob::CuTensorBlob{T})
  p = CudaRT.alloc_host(T, length(blob))
  p = convert(Ptr{T}, p)
  return pointer_to_array(p, size(blob))
end

function free_host_array(backend::GPUBackend, data::Array)
  CudaRT.free_host(convert(Ptr{Void}, pointer(data)))
end

function setup_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  return Array[alloc_host_array(backend, x) for x in state.blobs]
end

function shutdown_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  map(a -> free_host_array(backend, a), state.data_blocks)
end