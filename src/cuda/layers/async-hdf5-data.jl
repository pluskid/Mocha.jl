function alloc_host_array{T}(backend::GPUBackend, blob::CuTensorBlob{T})
  p = CUDA.cualloc_host(T, sizeof(blob))
  p = convert(Ptr{T}, p)
  return pointer_to_array(p, size(blob))
end

function free_host_array(backend::GPUBackend, data::Array{Any})
  CUDA.cualloc_free(convert(Ptr{Void}, pointer(data)))
end

function setup_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  return Array[alloc_host_array(backend, x) for x in state.blobs]
end

function shutdown_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  map(free_host_array, state.etc)
end