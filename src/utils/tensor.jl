export split_dims

# Split the dimension of a ND-tensor into 3 parts:
#   (dim_pre, dim_mid, dim_post)
function split_dims{T}(tensor::T, dim::Int)
  dims = size(tensor)
  dim_pre  = prod(dims[1:dim-1])
  dim_mid  = dims[dim]
  dim_post = prod(dims[dim+1:end])

  (dim_pre, dim_mid, dim_post)
end
