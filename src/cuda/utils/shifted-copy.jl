export copy_to_shifted!, copy_from_shifted!

function copy_to_shifted!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, shift::NTuple{2,Int})

  dims = size(src)
  dims2 = size(dst)

  shift_dim, shift_amount = shift

  dim_x  = prod(dims[1:shift_dim-1])
  @assert dim_x == prod(dims2[1:shift_dim-1])
  dim_y  = dims[shift_dim]
  dim_y2 = dims2[shift_dim]
  dim_z  = prod(dims[shift_dim+1:end])
  @assert dim_z == prod(dims2[shift_dim+1:end])

  x_block = round(Int64, ceil(float(dim_x)/CUDA.THREADS_PER_BLOCK_X))
  y_block = round(Int64, ceil(float(dim_y)/CUDA.THREADS_PER_BLOCK_Y))
  z_block = round(Int64, ceil(float(dim_z)/CUDA.THREADS_PER_BLOCK_Z))

  if T == Float32
    kernel = backend.mocha.copy_to_shifted_float
  elseif T == Float64
    kernel = backend.mocha.copy_to_shifted_double
  else
    error("Unsupported data type $T for shifted copy")
  end

  CUDA.launch(kernel, (x_block,y_block,z_block),
      (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
      (dst.ptr.p,src.ptr.p,dim_x,dim_y,dim_z,dim_y2,shift_amount))
end

function copy_from_shifted!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, shift::NTuple{2,Int})

  dims = size(dst)
  dims2 = size(src)

  shift_dim, shift_amount = shift

  dim_x  = prod(dims[1:shift_dim-1])
  @assert dim_x == prod(dims2[1:shift_dim-1])
  dim_y  = dims[shift_dim]
  dim_y2 = dims2[shift_dim]
  dim_z  = prod(dims[shift_dim+1:end])
  @assert dim_z == prod(dims2[shift_dim+1:end])

  x_block = round(Int64, ceil(float(dim_x)/CUDA.THREADS_PER_BLOCK_X))
  y_block = round(Int64, ceil(float(dim_y)/CUDA.THREADS_PER_BLOCK_Y))
  z_block = round(Int64, ceil(float(dim_z)/CUDA.THREADS_PER_BLOCK_Z))

  if T == Float32
    kernel = backend.mocha.copy_from_shifted_float
  elseif T == Float64
    kernel = backend.mocha.copy_from_shifted_double
  else
    error("Unsupported data type $T for shifted copy")
  end

  CUDA.launch(kernel, (x_block,y_block,z_block),
      (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
      (dst.ptr.p,src.ptr.p,dim_x,dim_y,dim_z,dim_y2,shift_amount))
end
