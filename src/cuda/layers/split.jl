function backward{N}(backend::GPUBackend, state::SplitLayerState{N}, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    diff = diffs[1]
    len = length(diff)
    one = convert(eltype(diff), 1)
    for i = 2:N
      CuBLAS.axpy(backend.cublas_ctx, len, one, state.blobs_diff[i].ptr, 1, diff.ptr, 1)
    end
  end
end

