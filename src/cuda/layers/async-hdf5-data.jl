#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function setup_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  return [CuHostBlob(backend, blob) for blob in state.blobs]
end

function shutdown_etc(backend::GPUBackend, state::AsyncHDF5DataLayerState)
  map(destroy, state.data_blocks)
end