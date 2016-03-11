#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function backward{N}(backend::GPUBackend, state::SplitLayerState{N}, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    diff = diffs[1]
    len = length(diff)
    one = convert(eltype(diff), 1)
    for i = 2:N
      CuBLAS.axpy(get_cublas_ctx(backend), len, one, get_ptr(state.blobs_diff[i]), 1, get_ptr(diff), 1)
    end
  end
end

