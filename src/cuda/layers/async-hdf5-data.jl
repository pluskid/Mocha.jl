function setup_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  return [CuHostBlob(backend, blob) for blob in state.blobs]
end

function shutdown_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  map(destroy, state.data_blocks)
end