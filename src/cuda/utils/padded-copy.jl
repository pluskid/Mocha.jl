export dense2padded!, padded2dense!

function dense2padded!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, pad_head::NTuple{2,Int}, pad_tail::NTuple{2,Int}, mirror::Bool)

    width, height, channels, num = size(src)
    chann_num = channels*num

    x_block = int(ceil(float(chann_num)/CUDA.THREADS_PER_BLOCK_X))
    y_block = int(ceil(float(width)/CUDA.THREADS_PER_BLOCK_Y))
    z_block = int(ceil(float(height)/CUDA.THREADS_PER_BLOCK_Z))

    if T == Float32
      kernel = backend.mocha.dense_to_padded_float
    elseif T == Float64
      kernel = backend.mocha.dense_to_padded_double
    else
      error("Unsupported data type $T for padded copy")
    end
    mirror = convert(Int, mirror)
    CUDA.launch(kernel, (x_block,y_block,z_block),
        (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
        (dst.ptr.p,src.ptr.p,width,height,pad_head[1],pad_head[2],pad_tail[1],pad_tail[2],chann_num,mirror))
end

function padded2dense!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, pad_head::NTuple{2,Int}, pad_tail::NTuple{2,Int}, mirror::Bool)

    width, height, channels, num = size(dst)
    chann_num = channels*num

    x_block = int(ceil(float(chann_num)/CUDA.THREADS_PER_BLOCK_X))
    y_block = int(ceil(float(width)/CUDA.THREADS_PER_BLOCK_Y))
    z_block = int(ceil(float(height)/CUDA.THREADS_PER_BLOCK_Z))

    if T == Float32
      kernel = backend.mocha.padded_to_dense_float
    elseif T == Float64
      kernel = backend.mocha.padded_to_dense_double
    else
      error("Unsupported data type $T for padded copy")
    end
    mirror = convert(Int, mirror)
    CUDA.launch(kernel, (x_block,y_block,z_block),
        (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z),
        (dst.ptr.p,src.ptr.p,width,height,pad_head[1],pad_head[2],pad_tail[1],pad_tail[2],chann_num, mirror))
end

function dense2padded!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, pad::NTuple{2,Int}, mirror=false)
  dense2padded!(backend, dst, src, pad, pad, mirror)
end
function padded2dense!{T}(backend::GPUBackend, dst::CuTensorBlob{T},
    src::CuTensorBlob{T}, pad::NTuple{2,Int}, mirror=false)
  padded2dense!(backend, dst, src, pad, pad, mirror)
end

