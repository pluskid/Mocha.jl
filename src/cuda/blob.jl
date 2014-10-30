abstract CuBlobDescriptor
type CuPODBlobDescriptor <: CuBlobDescriptor end
type CuTensorBlobDescriptor <: CuBlobDescriptor
  desc :: CuDNN.TensorDescriptor
end
type CuFilterBlobDescriptor <: CuBlobDescriptor
  desc :: CuDNN.FilterDescriptor
end

type CuTensorBlob{T<:FloatingPoint, T2<:CuBlobType} <: Blob
  ptr   :: CuPtr
  shape :: NTuple{4, Int}
  len   :: Int

  desc  :: CuBlobDescriptor

  CuTensorBlob{T<:FloatingPoint}(dtype::Type{T}, desc::CuBlobType, n::Int, c::Int, h::Int, w::Int) = begin
    len = n*c*h*w
    ptr = cualloc(dtype, len)
    return new(ptr, (w,h,c,n), len, desc)
  end
end
CuTensorBlob(dtype::Type; desc::CuBlobType=CuPODBlobDescriptor(), n::Int=1, c::Int=1, h::Int=1, w::Int=1) =
    CuTensorBlob(dtype, desc, n, c, h, w)

length(b::CuTensorBlob) = b.len
size(b::CuTensorBlob) = b.shape
eltype{T1,T2}(b::CuTensorBlob) = T1
