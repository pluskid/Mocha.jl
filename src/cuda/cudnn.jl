export CuDNN

module CuDNN
using ..CUDA

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
  elseif dtype == Float64
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

function set_tensor4d_descriptor{T<:FloatingPoint}(desc::Tensor4dDescriptor, dtype::Type{T}, dims :: NTuple{4, Int})
  w,h,c,n = dims
  @cudnncall(:cudnnSetTensor4dDescriptor, (Tensor4dDescriptor, Cint, Cint, Cint, Cint, Cint, Cint),
             desc, CUDNN_TENSOR_NCHW, cudnn_data_type(dtype), n, c, h, w)
end

function set_tensor4d_descriptor{T<:FloatingPoint}(desc::Tensor4dDescriptor, dtype::Type{T},
                                                   dims :: NTuple{4, Int}, stride :: NTuple{4, Int})
  w, h, c, n = dims
  wStride, hStride, cStride, nStride = stride
  @cudnncall(:cudnnSetTensor4dDescriptorEx, (Tensor4dDescriptor, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
             desc, cudnn_data_type(dtype), n,c,h,w,nStride,cStride,hStride,wStride)
end

function create_tensor4d_descriptor(dtype::Type, dims :: NTuple{4, Int})
  desc = create_tensor4d_descriptor()
  set_tensor4d_descriptor(desc, dtype, dims)
  return desc
end

function create_tensor4d_descriptor(dtype::Type, dims :: NTuple{4, Int}, stride :: NTuple{4, Int})
  desc = create_tensor4d_descriptor()
  set_tensor4d_descriptor(desc, dtype, dims, stride)
  return desc
end

function get_tensor4d_descriptor(desc::Tensor4dDescriptor)
  dtype = Cint[0]
  n = Cint[0]; c = Cint[0]; h = Cint[0]; w = Cint[0]
  nStride = Cint[0]; cStride = Cint[0]; hStride = Cint[0]; wStride = Cint[0]
  @cudnncall(:cudnnGetTensor4dDescriptor, (Tensor4dDescriptor, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                                           Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), desc, dtype, n,c,h,w,
                                           nStride, cStride, hStride, wStride)
  return (cudnn_data_type(dtype[1]), (w[1],h[1],c[1],n[1]), (wStride[1],hStride[1],cStride[1],nStride[1]))
end

function destroy_tensor4d_descriptor(desc :: Tensor4dDescriptor)
  @cudnncall(:cudnnDestroyTensor4dDescriptor, (Tensor4dDescriptor,), desc)
end

function transform_tensor4d(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @cudnncall(:cudnnTransformTensor4d, (Handle, Tensor4dDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, src_desc, src.p, dest_desc, dest.p)
end


# Tensor bias addition mode
const CUDNN_ADD_IMAGE   = 0     # add one image to every feature maps of each input
const CUDNN_ADD_SAME_HW = 0
const CUDNN_ADD_FEATURE_MAP = 1 # add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
const CUDNN_ADD_SAME_CHW    = 1
const CUDNN_ADD_SAME_C      = 2 # add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input
const CUDNN_ADD_FULL_TENSOR = 3 # add 2 tensors with same n,c,h,w

function add_tensor4d{T<:FloatingPoint}(handle::Handle, mode::Int, alpha::T,
                                        bias_desc::Tensor4dDescriptor, bias::CuPtr,
                                        srcdst_desc::Tensor4dDescriptor, srcdst::CuPtr)
  @assert CUDNN_ADD_IMAGE <= mode <= CUDNN_ADD_FULL_TENSOR
  @assert typeof(alpha) == get_tensor4d_descriptor(bias_desc)[1]
  alpha_ptr = T[alpha]

  @cudnncall(:cudnnAddTensor4d, (Handle, Cint, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
            handle, mode, alpha_ptr, bias_desc, bias.p, srcdst_desc, srcdst.p)
end

function set_tensor4d{T<:FloatingPoint}(handle::Handle, desc::Tensor4dDescriptor, data::CuPtr, val::T)
  @assert typeof(val) == get_tensor4d_descriptor(desc)[0]
  val_ptr = T[val]

  @cudnncall(:cudnnSetTensor4d, (Handle, Tensor4dDescriptor, Ptr{Void}, Ptr{Void}),
             handle, desc, data.p, val_ptr)
end


# Convolution Mode
const CUDNN_CONVOLUTION       = 0
const CUDNN_CROSS_CORRELATION = 1

# Convolution Path
const CUDNN_CONVOLUTION_FWD         = 0 # Tensor Convolution function
const CUDNN_CONVOLUTION_WEIGHT_GRAD = 1 # Weight Gradient update function
const CUDNN_CONVOLUTION_DATA_GRAD   = 2 # Data Gradient update function

function create_filter_descriptor()
  desc = FilterDescriptor[0]
  @cudnncall(:cudnnCreateFilterDescriptor, (Ptr{FilterDescriptor},), desc)
  return desc[1]
end
function set_filter_descriptor{T<:FloatingPoint}(desc::FilterDescriptor, dtype::Type{T}, dims :: NTuple{4, Int})
  w,h,c,k = dims
  @cudnncall(:cudnnSetFilterDescriptor, (FilterDescriptor, Cint, Cint, Cint, Cint, Cint),
             desc, cudnn_data_type(dtype), k, c, h, w)
           end
function create_filter_descriptor(dtype::Type, dims :: NTuple{4, Int})
  desc = create_filter_descriptor()
  set_filter_descriptor(desc, dtype, dims)
  return desc
end
function get_filter_descriptor(desc::FilterDescriptor)
  k = Cint[0]; c = Cint[0]; h = Cint[0]; w = Cint[0]
  dtype = Cint[0]
  @cudnncall(:cudnnGetFilterDescriptor, (FilterDescriptor,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
             desc, dtype, k, c, h, w)
  return (cudnn_data_type(dtype[1]), w[1], h[1], c[1], k[1])
end
function destroy_filter_descriptor(desc::FilterDescriptor)
  @cudnncall(:cudnnDestroyFilterDescriptor, (FilterDescriptor,), desc)
end

function create_convolution_descriptor()
  desc = ConvolutionDescriptor[0]
  @cudnncall(:cudnnCreateConvolutionDescriptor, (Ptr{ConvolutionDescriptor},), desc)
  return desc[1]
end
function set_convolution_descriptor(desc::ConvolutionDescriptor, input_desc::Tensor4dDescriptor,
    filter_desc::FilterDescriptor, pad::NTuple{2, Int}, stride::NTuple{2, Int}, upscale::NTuple{2, Int},
    conv_mode :: Int)

  @assert CUDNN_CONVOLUTION <= conv_mode CUDNN_CROSS_CORRELATION
  pad_w, pad_h = pad
  v, u = stride
  upscalex, upscaley = upscale

  @cudnncall(:cudnnSetConvolutionDescriptor, (ConvolutionDescriptor, Tensor4dDescriptor,
                                              FilterDescriptor, Cint, Cint, Cint, Cint,
                                              Cint, Cint, Cint),
             desc, input_desc, filter_desc, pad_h, pad_w, u, v, upscalex, upscaley, conv_mode)
end

function create_convolution_descriptor(input_desc::Tensor4dDescriptor,
    filter_desc::FilterDescriptor, pad::NTuple{2, Int}, stride::NTuple{2, Int}, upscale::NTuple{2, Int},
    conv_mode :: Int)
  desc = create_convolution_descriptor()
  set_convolution_descriptor(desc, input_desc, filter_desc, pad, stride, upscale, conv_mode)
  return desc
end

function set_convolution_descriptor_ex(desc::ConvolutionDescriptor, dims::NTuple{4, Int},
    n_filter::Int, kernel_wh::NTuple{2, Int}, pad::NTuple{2, Int}, stride::NTuple{2, Int},
    upscale::NTuple{2, Int}, conv_mode::Int)

  @assert CUDNN_CONVOLUTION <= conv_mode CUDNN_CROSS_CORRELATION
  w,h,c,n = dims
  s,r = kernel_wh
  pad_w, pad_h = pad
  v,u = stride
  upscalex, upscaley = upscale

  @cudnncall(:cudnnSetConvolutionDescriptorEx, (ConvolutionDescriptor, Cint,Cint,Cint,Cint,
                                                Cint,Cint,Cint, Cint,Cint, Cint,Cint, Cint,Cint, Cint),
             desc, n,c,h,w, n_filter, r,s, pad_h,pad_w, u,v, upscalex,upscaley, conv_mode)
end
function destroy_convolution_descriptor(desc::ConvolutionDescriptor)
  @cudnncall(:cudnnDestroyConvolutionDescriptor, (ConvolutionDescriptor,), desc)
end

function get_output_tensor4d_dim(desc::ConvolutionDescriptor, path::Int)
  @assert CUDNN_CONVOLUTION_FWD <= path <= CUDNN_CONVOLUTION_DATA_GRAD
  n = Cint[0]; c = Cint[0]; h = Cint[0]; w = Cint[0]
  @cudnncall(:cudnnGetOutputTensor4dDim, (ConvolutionDescriptor, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}),
             desc, path, n, c, h, w)
  return (w[1], h[1], c[1], n[1])
end

# accumulate the result of the operation into output buffer (or overwrite) ?
const CUDNN_RESULT_ACCUMULATE      = 0 # Evaluate O += I * F
const CUDNN_RESULT_NO_ACCUMULATE   = 1 # Evaluate O = I * F

function convolution_forward(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr,
    filter_desc::FilterDescriptor, filter::CuPtr, conv::ConvolutionDescriptor,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, mode::Int)

  @assert CUDNN_RESULT_ACCUMULATE <= mode <= CUDNN_RESULT_NO_ACCUMULATE
  @cudnncall(:cudnnConvolutionForward, (Handle, Tensor4dDescriptor, Ptr{Void},
                                        FilterDescriptor, Ptr{Void}, ConvolutionDescriptor,
                                        Tensor4dDescriptor, Ptr{Void}, Cint),
             handle, src_desc, src.p, filter_desc, filter.p, conv, dest_desc, dest.p, mode)
end

function colvolution_backward_bias(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, mode::Int)
  @assert CUDNN_RESULT_ACCUMULATE <= mode <= CUDNN_RESULT_NO_ACCUMULATE
  @cudnncall(:cudnnConvolutionBackwardBias, (Handle, Tensor4dDescriptor, Ptr{Void},
                                             Tensor4dDescriptor, Ptr{Void}, Cint),
             handle, src_desc, src.p, dest_desc, dest.p, mode)
end

function convolution_backward_filter(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr,
    diff_desc::Tensor4dDescriptor, diff::CuPtr, conv::ConvolutionDescriptor,
    grad_desc::FilterDescriptor, grad::CuPtr, mode::Int)
  @assert CUDNN_RESULT_ACCUMULATE <= mode <= CUDNN_RESULT_NO_ACCUMULATE
  @cudnncall(:cudnnConvolutionBackwardFilter, (Handle, Tensor4dDescriptor, Ptr{Void},
                                               Tensor4dDescriptor, Ptr{Void},
                                               ConvolutionDescriptor,
                                               FilterDescriptor, Ptr{Void}, Cint),
             handle, src_desc, src.p, diff_desc, diff.p, conv, grad_desc, grad.p, mode)
end

function convolution_backward_data(handle::Handle, filter_desc::FilterDescriptor, filter::CuPtr,
    diff_desc::Tensor4dDescriptor, diff::CuPtr, conv::ConvolutionDescriptor,
    grad_desc::Tensor4dDescriptor, grad::CuPtr, mode::Int)
  @assert CUDNN_RESULT_ACCUMULATE <= mode <= CUDNN_RESULT_NO_ACCUMULATE
  @cudnncall(:cudnnConvolutionForwardData, (Handle, FilterDescriptor, Ptr{Void},
                                            Tensor4dDescriptor, Ptr{Void},
                                            ConvolutionDescriptor, Tensor4dDescriptor,
                                            Ptr{Void}, Cint),
             handle, filter_desc, filter.p, diff_desc, diff.p, conv, grad_desc, grad.p, mode)
end


const CUDNN_SOFTMAX_FAST     = 0  # straightforward implementation
const CUDNN_SOFTMAX_ACCURATE = 1  # subtract max from every point to avoid overflow

const CUDNN_SOFTMAX_MODE_INSTANCE = 0 # compute the softmax over all C, H, W for each N
const CUDNN_SOFTMAX_MODE_CHANNEL = 1  # compute the softmax over all C for each H, W, N

function softmax_forward(handle::Handle, algorithm::Int, mode::Int,
    src_desc::Tensor4dDescriptor, src::CuPtr, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @assert CUDNN_SOFT_MAX_FAST <= algorithm <= CUDNN_SOFTMAX_ACCURATE
  @assert CUDNN_SOFTMAX_MODE_INSTANCE <= mode <= CUDNN_SOFTMAX_MODE_CHANNEL
  @cudnncall(:cudnnSoftmaxForward, (Handle, Cint, Cint, Tensor4dDescriptor, Ptr{Void},
                                    Tensor4dDescriptor, Ptr{Void}),
             handle, algorithm, mode, src_desc, src.p, dest_desc, dest.p)
end

function softmax_backward(handle::Handle, algorithm::Int, mode::Int,
    src_desc::Tensor4dDescriptor, src::CuPtr, srcdiff_desc::Tensor4dDescriptor, srcdiff::CuPtr,
    destdiff_desc::Tensor4dDescriptor, descdiff::CuPtr)
  @assert CUDNN_SOFT_MAX_FAST <= algorithm <= CUDNN_SOFTMAX_ACCURATE
  @assert CUDNN_SOFTMAX_MODE_INSTANCE <= mode <= CUDNN_SOFTMAX_MODE_CHANNEL
  @cudnncall(:cudnnSoftmaxBackward, (Handle, Cint, Cint, Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, algorithm, mode, src_desc, src.p, srcdiff_desc, srcdiff.p,
             destdiff_desc, destdiff.p)
end

const CUDNN_POOLING_MAX     = 0
const CUDNN_POOLING_AVERAGE = 1

function create_pooling_descriptor()
  desc = PoolingDescriptor[0]
  @cudnncall(:cudnnCreatePoolingDescriptor, (Ptr{PoolingDescriptor},), desc)
  return desc[1]
end
function set_pooling_descriptor(desc::PoolingDescriptor, mode::Int, dims::NTuple{2, Int}, stride::NTuple{2, Int})
  @assert CUDNN_POOLING_MAX <= mode <= CUDNN_POOLING_AVERAGE
  w,h = dims
  stride_w, stride_h = stride
  @cudnncall(:cudnnSetPoolingDescriptor, (PoolingDescriptor, Cint, Cint,Cint,Cint,Cint),
             desc, mode, h,w, stride_h, stride_w)
end
function get_pooling_descriptor(desc::PoolingDescriptor)
  mode = Cint[0]
  h = Cint[0]; w = Cint[0]; stride_h = Cint[0]; stride_w = Cint[0]
  @cudnncall(:cudnGetPoolingDescriptor, (PoolingDescriptor, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                                         Ptr{Cint}, Ptr{Cint}),
             desc, mode, h, w, stride_h, stride_w)
  return (mode[1], (w,h), (stride_w, stride_h))
end
function destroy_pooling_descriotpr(desc::PoolingDescriptor)
  @cudnncall(:cudnnDestroyPoolingDescroptor, (PoolingDescriptor,), desc)
end

function pooling_forward(handle::Handle, pooling::PoolingDescriptor,
    src_desc::Tensor4dDescriptor, src::CuPtr, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @cudnncall(:cudnnPoolingForward, (Handle, PoolingDescriptor, Tensor4dDescriptor, Ptr{Void},
                                    Tensor4dDescriptor, Ptr{Void}),
             handle, pooling, src_desc, src.p, dest_desc, dest.p)
end
function pooling_backward(handle::Handle, pooling::PoolingDescriptor,
    src_desc::Tensor4dDescriptor, src::CuPtr, srcdiff_desc::Tensor4dDescriptor, srcdiff::CuPtr,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, destdiff_desc::Tensor4dDescriptor, destdiff::CuPtr)
  @cudnncall(:cudnnPoolingBackward, (Handle, PoolingDescriptor, Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void},Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void}),
             handle, pooling, src_desc, src.p, srcdiff_desc, srcdiff.p,
             dest_desc, dest.p, destdiff_desc, destdiff.p)
end


const CUDNN_ACTIVATION_SIGMOID = 0
const CUDNN_ACTIVATION_RELU    = 1
const CUDNN_ACTIVATION_TANH    = 2

function activation_forward(handle::Handle, mode::Int, src_desc::Tensor4dDescriptor, src::CuPtr,
    dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @assert CUDNN_ACTIVATION_SIGMOID <= mode <+ CUDNN_ACTIVATION_TANH
  @cudnncall(:cudnnActivationForward, (Handle, Cint, Tensor4dDescriptor, Ptr{Void},
                                       Tensor4dDescriptor, Ptr{Void}),
             handle, mode, src_desc, src.p, dest_desc, dest.p)
end
function activation_backward(handle::Handle, mode::Int,
    src_desc::Tensor4dDescriptor, src::CuPtr, srcdiff_desc::Tensor4dDescriptor, srcdiff::CuPtr,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, destdiff_desc::Tensor4dDescriptor, destdiff::CuPtr)
  @assert CUDNN_ACTIVATION_SIGMOID <= mode <+ CUDNN_ACTIVATION_TANH
  @cudnncall(:cudnnActivationBackward, (Handle, Cint, Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void},Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void}),
             handle, mode, src_desc, src.p, srcdiff_desc, srcdiff.p,
             dest_desc, dest.p, destdiff_desc, destdiff.p)
end

end # module
