module CuDNN
using CUDA

# cudnnStatus_t
const  CUDNN_STATUS_SUCCESS          = 0
const  CUDNN_STATUS_NOT_INITIALIZED  = 1
const  CUDNN_STATUS_ALLOC_FAILED     = 2
const  CUDNN_STATUS_BAD_PARAM        = 3
const  CUDNN_STATUS_INTERNAL_ERROR   = 4
const  CUDNN_STATUS_INVALID_VALUE    = 5
const  CUDNN_STATUS_ARCH_MISMATCH    = 6
const  CUDNN_STATUS_MAPPING_ERROR    = 7
const  CUDNN_STATUS_EXECUTION_FAILED = 8
const  CUDNN_STATUS_NOT_SUPPORTED    = 9
const  CUDNN_STATUS_LICENSE_ERROR    = 10

immutable CuDNNError <: Exception
  code :: Int
end
const cudnn_error_description = [
  CUDNN_STATUS_SUCCESS          => "Success",
  CUDNN_STATUS_NOT_INITIALIZED  => "Not initialized",
  CUDNN_STATUS_ALLOC_FAILED     => "Alloc failed",
  CUDNN_STATUS_BAD_PARAM        => "Bad param",
  CUDNN_STATUS_INTERNAL_ERROR   => "Internal error",
  CUDNN_STATUS_INVALID_VALUE    => "Invalid value",
  CUDNN_STATUS_ARCH_MISMATCH    => "Arch mismatch",
  CUDNN_STATUS_MAPPING_ERROR    => "Mapping error",
  CUDNN_STATUS_EXECUTION_FAILED => "Execution failed",
  CUDNN_STATUS_NOT_SUPPORTED    => "Not supported",
  CUDNN_STATUS_LICENSE_ERROR    => "License error"
]
import Base.show
show(io::IO, error::CuDNNError) = print(io, cudnn_error_description[error.code])

macro cudnncall(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), "libcudnn"), Cint, $argtypes, $(args...)  )
    if int(_curet) != CUDNN_STATUS_SUCCESS
      throw(CuDNNError(int(_curet)))
    end
  end
end

typealias Handle Ptr{Void}
typealias StreamHandle Ptr{Void}

function create()
  handle = Handle[0]
  @cudnncall(:cudnnCreate, (Ptr{Handle},), handle)
  return handle[1]
end
function destroy(handle :: Handle)
  @cudnncall(:cudnnDestroy, (Handle,), handle)
end
function set_stream(handle::Handle, stream::StreamHandle)
  @cudnncall(:cudnnSetStream, (Handle, StreamHandle), handle, stream)
end
function get_stream(handle::Handle)
  s_handle = StreamHandle[0]
  @cudnncall(:cudnnGetStream, (Handle, Ptr{StreamHandle}), handle, s_handle)
  return s_handle[1]
end

# Data structures to represent Image/Filter and the Neural Network Layer
typealias Tensor4dDescriptor Ptr{Void}
typealias ConvolutionDescriptor Ptr{Void}
typealias PoolingDescriptor Ptr{Void}
typealias FilterDescriptor Ptr{Void}

const CUDNN_DATA_FLOAT = 0
const CUDNN_DATA_DOUBLE = 1
function cudnn_data_type{T<:FloatingPoint}(dtype::Type{T})
  if dtype == Float32
    return CUDNN_DATA_FLOAT
  elseif dtype == FLOAT64
    return CUDNN_DATA_DOUBLE
  else
    error("Unsupported data type $(dtype)")
  end
end
function cudnn_data_type(dtype :: Cint)
  if dtype == CUDNN_DATA_FLOAT
    return Float32
  elseif dtype == CUDNN_DATA_DOUBLE
    return Float64
  else
    error("Unknown CuDNN data code: $(dtype)")
  end
end

const CUDNN_TENSOR_NCHW = 0    # row major (wStride = 1, hStride = w)
const CUDNN_TENSOR_NHWC = 1    # feature maps interleaved ( cStride = 1 )

function create_tensor4d_descriptor()
  desc = Tensor4dDescriptor[0]
  @cudnncall(:cudnnCreateTensor4dDescriptor, (Tensor4dDescriptor,), desc)
  return desc[1]
end

function set_tensor4d_descriptor{T<:FloatingPoint}(desc::Tensor4dDescriptor, dtype::Type{T}, dims :: Vector{Int})
  n,c,h,w = dims
  @cudnncall(:cudnnSetTensor4dDescriptor, (Tensor4dDescriptor, Cint, Cint, Cint, Cint, Cint, Cint), 
             desc, CUDNN_TENSOR_NCHW, cudnn_data_type(dtype), n, c, h, w)
end

function set_tensor4d_descriptor{T<:FloatingPoint}(desc::Tensor4dDescriptor, dtype::Type{T}, 
                                                   dims :: Vector{Int}, stride :: Vector{Int})
  n, c, h, w = dims
  nStride, cStride, hStride, wStride = stride
  @cudnncall(:cudnnSetTensor4dDescriptorEx, (Tensor4dDescriptor, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
             desc, CUDNN_TENSOR_NCHW, cudnn_data_type(dtype), n,c,h,w,nStride,cStride,hStride,wStride)
end

function get_tensor4d_descriptor(desc::Tensor4dDescriptor)
  dtype = Cint[0]
  n = Cint[0]; c = Cint[0]; h = Cint[0]; w = Cint[0]
  nStride = Cint[0], cStride = Cint[0], hStride = Cint[0], wStride = Cint[0]
  @cudnncall(:cudnnGetTensor4dDescriptor, (Tensor4dDescriptor, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                                           Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), desc, dtype, n,c,h,w,
                                           nStride, cStride, hStride, wStride)
  return (cudnn_data_type(dtype[1]), (n[1],c[1],h[1],w[1]), (nStride[1],cStride[1],hStride[1],wStride[1]))
end

function destroy_tensor4d_descriptor(desc :: Tensor4dDescriptor)
  @cudnncall(:cudnnDestroyTensor4dDescriptor, (Tensor4dDescriptor,), desc)
end

function transform_tensor4d(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @cudnncall(:cudnnTransformTensor4d, (Handle, Tensor4dDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}), 
             handle, src_desc, src.p, dest_desc, dest.p)
end
