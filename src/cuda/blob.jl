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
function CuTensorBlob{T<:FloatingPoint}(dtype::Type{T}, desc::CuBlobDescriptor, n::Int, c::Int, h::Int, w::Int)
  len = n*c*h*w
  ptr = CUDA.cualloc(dtype, len)
  return CuTensorBlob{T}(ptr, (w,h,c,n), len, desc)
end
CuTensorBlob(dtype::Type; desc::CuBlobDescriptor=CuPODBlobDescriptor(), n::Int=1, c::Int=1, h::Int=1, w::Int=1) =
    CuTensorBlob(dtype, desc, n, c, h, w)

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape
eltype{T1}(b::CuTensorBlob{T1}) = T1

function copy!{T}(dst :: CuTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  CuBLAS.set_vector(vec(src), dst.ptr)
end
function copy!{T}(dst :: Array{T}, src :: CuTensorBlob{T})
  @assert length(dst) == length(src)
  CuBLAS.get_vector(src.ptr, dst)
end
function fill!{T}(dst :: CuTensorBlob{T}, val)
  val = convert(T, val)
  val = convert(Ptr{Void}, T[val])
  CuBLAS.set_vector(length(dst), sizeof(T), val, 0, dst.ptr, 1)
end
