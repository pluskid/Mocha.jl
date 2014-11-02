export CuBlobDescriptor, CuPODBlobDescriptor, CuTensorBlobDescriptor, CuFilterBlobDescriptor
export CuTensorBlob

abstract CuBlobDescriptor
type CuPODBlobDescriptor <: CuBlobDescriptor end
type CuTensorBlobDescriptor <: CuBlobDescriptor
  desc :: CuDNN.Tensor4dDescriptor
end
type CuFilterBlobDescriptor <: CuBlobDescriptor
  desc :: CuDNN.FilterDescriptor
end

type CuTensorBlob{T<:FloatingPoint} <: Blob
  ptr   :: CuPtr
  shape :: NTuple{4, Int}
  len   :: Int

  desc  :: CuBlobDescriptor
end
function CuTensorBlob{T<:FloatingPoint}(dtype::Type{T}, desc::CuBlobDescriptor, w::Int, h::Int, c::Int, n::Int)
  len = n*c*h*w
  ptr = CUDA.cualloc(dtype, len)
  return CuTensorBlob{T}(ptr, (w,h,c,n), len, desc)
end
CuTensorBlob(dtype::Type; desc::CuBlobDescriptor=CuPODBlobDescriptor(), w::Int=1, h::Int=1, c::Int=1, n::Int=1) =
    CuTensorBlob(dtype, desc, w, h, c, n)

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape
eltype{T1}(b::CuTensorBlob{T1}) = T1

function copy!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CuBLAS.set_vector(src, dst.ptr)
end
function copy!{T}(dst :: Array{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CuBLAS.get_vector(src.ptr, dst)
end
function copy!{T}(dst :: CuTensorBlob{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  @CUDA.cucall(:cuMemcpy, (Ptr{Void}, Ptr{Void}, Cint), dst.ptr.p, src.ptr.p, length(dst)*sizeof(T))
end
function fill!{T}(dst :: CuTensorBlob{T}, val)
  val_vec = Array(T, length(dst))
  fill!(val_vec, val)
  copy!(dst, val_vec)
end
#function erase!{T}(dst :: CuTensorBlob{T})
#  @CUDA.cucall(:cuMemsetD8, (Ptr{Void}, Cuchar, Csize_t), dst.ptr.p, 0, length(dst)*sizeof(T))
#end

# Get canonical 4D tensor dims
# Note we store data in column-major order, thus
# the data shape is width, height, channel, num
function get_nchw_dims(dims...)
  if length(dims) > 4
    error("Tensor dimension ($(length(dims))) twoo high, only 4D tensor supported")
  end
  if length(dims) < 4
    # add singleton dimensions
    dims = tuple(ones(Int, 4-length(dims))..., dims...)
  end
  return dims
end

function cudnn_make_tensor_blob(dtype::Type, dims...)
  desc = CuDNN.create_tensor4d_descriptor()
  dims = get_nchw_dims(dims...)
  CuDNN.set_tensor4d_descriptor(desc, dtype, dims)
  return CuTensorBlob(dtype, CuTensorBlobDescriptor(desc), dims...)
end
function cudnn_make_pod_blob(dtype::Type, dims...)
  dims = get_nchw_dims(dims...)
  return CuTensorBlob(dtype, CuPODBlobDescriptor(), dims...)
end
