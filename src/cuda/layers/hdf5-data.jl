function set_blob_data{T}(data::Array{T}, blob::CuTensorBlob{T}, blob_idx::Int)
  n_fea = prod(size(data)[1:3])
  ptr = convert(Ptr{Void}, blob.ptr.p) + sizeof(T) * n_fea * (blob_idx-1) # note 0-based indexing in CUDA Vector
  CuBLAS.set_vector(length(data), sizeof(T), convert(Ptr{Void},data), 1, ptr, 1)
end

