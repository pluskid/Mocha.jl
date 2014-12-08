export copy_to_shifted!, copy_from_shifted!

function copy_to_shifted!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, shifts::Vector{Int})

  width, height, channels, num = size(src)
  width2, height2, channels2, num2 = size(dst)
  chann_num = channels*num

  x_block = int(ceil(float(chann_num)/CUDA.THREADS_PER_BLOCK_X))
  y_block = int(ceil(float(width)/CUDA.THREADS_PER_BLOCK_Y))
  z_block = int(ceil(float(height)/CUDA.THREADS_PER_BLOCK_Z))

  if T == Float32
    kernel = backend.mocha.copy_to_shifted_float
  elseif T == Float64
    kernel = backend.mocha.copy_to_shifted_double
  else
    error("Unsupported data type $T for shifted copy")
  end

  CUDA.launch(kernel, (x_block,y_block,z_block),
      (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
      (dst.ptr.p,src.ptr.p,width,height,channels,num,
      shifts[1],shifts[2],shifts[3],shifts[4],
      width2,height2,channels2))
end

function copy_from_shifted!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, shifts::Vector{Int})

  width, height, channels, num = size(dst)
  width2, height2, channels2, num2 = size(src)
  chann_num = channels*num

  x_block = int(ceil(float(chann_num)/CUDA.THREADS_PER_BLOCK_X))
  y_block = int(ceil(float(width)/CUDA.THREADS_PER_BLOCK_Y))
  z_block = int(ceil(float(height)/CUDA.THREADS_PER_BLOCK_Z))

  if T == Float32
    kernel = backend.mocha.copy_from_shifted_float
  elseif T == Float64
    kernel = backend.mocha.copy_from_shifted_double
  else
    error("Unsupported data type $T for shifted copy")
  end

  CUDA.launch(kernel, (x_block,y_block,z_block),
      (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
      (dst.ptr.p,src.ptr.p,width,height,channels,num,
      shifts[1],shifts[2],shifts[3],shifts[4],
      width2,height2,channels2))
end
