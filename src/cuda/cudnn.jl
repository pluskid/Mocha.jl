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
using Compat
const cudnn_error_description = @compat(Dict(
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
))
import Base.show
show(io::IO, error::CuDNNError) = print(io, cudnn_error_description[error.code])

if is_windows()
  const libcudnn = Libdl.find_library(["cudnn64_5.dll", "cudnn64_80.dll", "cudnn64_70.dll", "cudnn32_80.dll",
                                       "cudnn32_70.dll"], [""])
  @assert (libcudnn != "") "Could not find a CUDA neural net DLL [cudnn64_80.dll, cudnn64_70.dll, cudnn32_80.dll, cudnn32_70.dll]. See: http://mochajl.readthedocs.io/en/latest/user-guide/backend.html#cuda-backend"
else
  const libcudnn = Libdl.find_library(["libcudnn"], [""])
  @assert (libcudnn != "") "Could not find CUDA neural shared library [libcudnn]. See http://mochajl.readthedocs.io/en/latest/user-guide/backend.html#cuda-backend"
end

macro cudnncall(fv, argtypes, args...)
  args = map(esc, args)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $libcudnn), Cint, $argtypes, $(args...)  )
    if round(Int, _curet) != CUDNN_STATUS_SUCCESS
      throw(CuDNNError(round(Int, _curet)))
    end
  end
end

const Handle = Ptr{Void}
const StreamHandle = Ptr{Void}

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
const Tensor4dDescriptor = Ptr{Void}
const ConvolutionDescriptor = Ptr{Void}
const PoolingDescriptor = Ptr{Void}
const FilterDescriptor = Ptr{Void}

const CUDNN_DATA_FLOAT = 0
const CUDNN_DATA_DOUBLE = 1
function cudnn_data_type{T<:AbstractFloat}(dtype::Type{T})
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

# cudnnNanPropagation_t
const CUDNN_NOT_PROPAGATE_NAN = 0
const CUDNN_PROPAGATE_NAN = 1

function create_tensor4d_descriptor()
  desc = Tensor4dDescriptor[0]
  @cudnncall(:cudnnCreateTensorDescriptor, (Tensor4dDescriptor,), desc)
  return desc[1]
end

function set_tensor4d_descriptor{T<:AbstractFloat}(desc::Tensor4dDescriptor, dtype::Type{T}, dims :: NTuple{4, Int})
  w,h,c,n = dims
  @cudnncall(:cudnnSetTensor4dDescriptor, (Tensor4dDescriptor, Cint, Cint, Cint, Cint, Cint, Cint),
             desc, CUDNN_TENSOR_NCHW, cudnn_data_type(dtype), n, c, h, w)
end

function set_tensor4d_descriptor{T<:AbstractFloat}(desc::Tensor4dDescriptor, dtype::Type{T},
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
  @cudnncall(:cudnnDestroyTensorDescriptor, (Tensor4dDescriptor,), desc)
end

function transform_tensor4d(handle::Handle, src_desc::Tensor4dDescriptor, src::CuPtr, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @cudnncall(:cudnnTransformTensor4d, (Handle, Tensor4dDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, src_desc, src.p, dest_desc, dest.p)
end


function add_tensor{T<:AbstractFloat}(handle::Handle, alpha::T,
                                        bias_desc::Tensor4dDescriptor, bias::CuPtr,
                                        beta::T,
                                        srcdst_desc::Tensor4dDescriptor, srcdst::CuPtr)
  @assert typeof(alpha) == get_tensor4d_descriptor(bias_desc)[1]
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]

  @cudnncall(:cudnnAddTensor, (Handle, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
            handle, alpha_ptr, bias_desc, bias.p, beta_ptr, srcdst_desc, srcdst.p)
end

function set_tensor4d{T<:AbstractFloat}(handle::Handle, desc::Tensor4dDescriptor, data::CuPtr, val::T)
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

# cudnn tensor format
const CUDNN_TENSOR_NCHW = 0  # row major (wStride = 1, hStride = w)
const CUDNN_TENSOR_NHWC = 1  # feature maps interleaved ( cStride = 1 )

function create_filter_descriptor()
  desc = FilterDescriptor[0]
  @cudnncall(:cudnnCreateFilterDescriptor, (Ptr{FilterDescriptor},), desc)
  return desc[1]
end
function set_filter_descriptor{T<:AbstractFloat}(desc::FilterDescriptor, dtype::Type{T}, dims :: NTuple{4, Int})
  w,h,c,k = dims
  @cudnncall(:cudnnSetFilter4dDescriptor, (FilterDescriptor, Cint, Cint, Cint, Cint, Cint, Cint),
             desc, cudnn_data_type(dtype), CUDNN_TENSOR_NCHW, k, c, h, w)
           end
function create_filter_descriptor(dtype::Type, dims :: NTuple{4, Int})
  desc = create_filter_descriptor()
  set_filter_descriptor(desc, dtype, dims)
  return desc
end
function get_filter_descriptor(desc::FilterDescriptor)
  k = Cint[0]; c = Cint[0]; h = Cint[0]; w = Cint[0]
  dtype = Cint[0]; tensor_format = Cint[0]
  @cudnncall(:cudnnGetFilterDescriptor, (FilterDescriptor,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
             desc, dtype, tensor_format, k, c, h, w)
  @assert tensor_format[1] == CUDNN_TENSOR_NCHW
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

  @assert CUDNN_CONVOLUTION <= conv_mode <= CUDNN_CROSS_CORRELATION
  pad_w, pad_h = pad
  v, u = stride
  upscalex, upscaley = upscale

  @cudnncall(:cudnnSetConvolution2dDescriptor, (ConvolutionDescriptor, Cint, Cint, Cint, Cint,
                                              Cint, Cint, Cint),
             desc, pad_h, pad_w, u, v, upscalex, upscaley, conv_mode)
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

  @assert CUDNN_CONVOLUTION <= conv_mode <= CUDNN_CROSS_CORRELATION
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

const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMPT_GEMM = 1
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4

const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2

function get_convolution_forward_algorithm(handle::Handle, src_desc::Tensor4dDescriptor,
    filter_desc::FilterDescriptor, conv_desc::ConvolutionDescriptor, dest_desc::Tensor4dDescriptor,
    preference::Int, mem_limit_bytes::Int)

  @assert CUDNN_CONVOLUTION_FWD_NO_WORKSPACE <= preference <= CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
  algor = Int[0]
  @cudnncall(:cudnnGetConvolutionForwardAlgorithm, (Handle, Tensor4dDescriptor, FilterDescriptor,
                                                    ConvolutionDescriptor, Tensor4dDescriptor, Int,
                                                    Csize_t, Ptr{Int}),
      handle, src_desc, filter_desc, conv_desc, dest_desc, preference, mem_limit_bytes, algor)
  algor = algor[1]
  @assert CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM <= algor <= CUDNN_CONVOLUTION_FWD_ALGO_FFT
  return algor
end

function get_convolution_forward_workspace_size(handle::Handle, src_desc::Tensor4dDescriptor,
    filter_desc::FilterDescriptor, conv_desc::ConvolutionDescriptor, dest_desc::Tensor4dDescriptor,
    algor::Int)

  @assert CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM <= algor <= CUDNN_CONVOLUTION_FWD_ALGO_FFT
  ws_size = Csize_t[0]
  @cudnncall(:cudnnGetConvolutionForwardWorkspaceSize, (Handle, Tensor4dDescriptor, FilterDescriptor,
                                                    ConvolutionDescriptor, Tensor4dDescriptor, Int, Ptr{Csize_t}),
      handle, src_desc, filter_desc, conv_desc, dest_desc, algor, ws_size)
  ws_size = ws_size[1]
  return ws_size
end


function convolution_forward{T<:AbstractFloat}(handle::Handle, alpha::T, src_desc::Tensor4dDescriptor, src::CuPtr,
    filter_desc::FilterDescriptor, filter::CuPtr, conv::ConvolutionDescriptor,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, workspace::CuPtr, workspace_size, algo::Int,
                             beta::T)
  #no workspace needed since we will use CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @assert CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM <= algo <= CUDNN_CONVOLUTION_FWD_ALGO_FFT
  @cudnncall(:cudnnConvolutionForward, (Handle, Ptr{Void}, Tensor4dDescriptor, Ptr{Void},
                                        FilterDescriptor, Ptr{Void}, ConvolutionDescriptor,
                                        Cint, Ptr{Void}, Csize_t, Ptr{Void},
                                        Tensor4dDescriptor, Ptr{Void}),
             handle, alpha_ptr, src_desc, src.p,
             filter_desc, filter.p, conv,
             algo, workspace.p, workspace_size, beta_ptr,
             dest_desc, dest.p)
end

function convolution_backward_bias{T<:AbstractFloat}(handle::Handle, alpha::T, src_desc::Tensor4dDescriptor, src::CuPtr,
    beta::T, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @cudnncall(:cudnnConvolutionBackwardBias, (Handle, Ptr{Void}, Tensor4dDescriptor, Ptr{Void},
      Ptr{Void}, Tensor4dDescriptor, Ptr{Void}), handle, alpha_ptr, src_desc, src.p, beta_ptr, dest_desc, dest.p)
end
function convolution_backward_filter{T<:AbstractFloat}(handle::Handle, alpha::T, src_desc::Tensor4dDescriptor, src::CuPtr,
    diff_desc::Tensor4dDescriptor, diff::CuPtr, conv::ConvolutionDescriptor,
    beta::T, grad_desc::FilterDescriptor, grad::CuPtr)
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]

  # XXX: properly query algorithm and allocate workspace
  bwd_filter_algor = 0
  workspace = C_NULL
  workspace_size = 0

  @cudnncall(:cudnnConvolutionBackwardFilter, 
             (Handle, 
              Ptr{Void},             # const void *alpha
              Tensor4dDescriptor,    # const cudnnTensorDescriptor_t xDesc
              Ptr{Void},             # const void *x
              Tensor4dDescriptor,    # const cudnnTensorDescroptor_t dyDesc
              Ptr{Void},             # const void *dy
              ConvolutionDescriptor, # const cudnnConvolutionDescriptor_t 
              Cint,                  # cudnnConvolutionBwdFilterAlgo_t
              Ptr{Void},             # void *workSpace
              Csize_t,               # size_t workSpaceSizeInBytes
              Ptr{Void},             # const void *beta
              FilterDescriptor,      # const cudnnFilterDescriptor_t dwDesc
              Ptr{Void}),            # void *dw
             handle, alpha_ptr, src_desc, src.p, diff_desc, diff.p, conv, 
             bwd_filter_algor, workspace, workspace_size,
             beta_ptr, grad_desc, grad.p)
end

function convolution_backward_data{T<:AbstractFloat}(handle::Handle, alpha::T, filter_desc::FilterDescriptor, filter::CuPtr,
    diff_desc::Tensor4dDescriptor, diff::CuPtr, conv::ConvolutionDescriptor,
    beta::T, grad_desc::Tensor4dDescriptor, grad::CuPtr)
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]

  # XXX: properly query algorithm and allocate workspace
  bwd_data_algor = 0
  workspace = C_NULL
  workspace_size = 0

  @cudnncall(:cudnnConvolutionBackwardData, (Handle, Ptr{Void}, FilterDescriptor, Ptr{Void},
                                            Tensor4dDescriptor, Ptr{Void},
                                            ConvolutionDescriptor,
                                            Cint, Ptr{Void}, Csize_t,
                                            Ptr{Void},Tensor4dDescriptor,
                                            Ptr{Void}),
             handle, alpha_ptr, filter_desc, filter.p, diff_desc, diff.p, conv,
             bwd_data_algor, workspace, workspace_size,
             beta_ptr, grad_desc, grad.p)
end


const CUDNN_SOFTMAX_FAST     = 0  # straightforward implementation
const CUDNN_SOFTMAX_ACCURATE = 1  # subtract max from every point to avoid overflow

const CUDNN_SOFTMAX_MODE_INSTANCE = 0 # compute the softmax over all C, H, W for each N
const CUDNN_SOFTMAX_MODE_CHANNEL = 1  # compute the softmax over all C for each H, W, N

function softmax_forward{T<:AbstractFloat}(handle::Handle, algorithm::Int, mode::Int,
    alpha::T, src_desc::Tensor4dDescriptor, src::CuPtr, beta::T, dest_desc::Tensor4dDescriptor, dest::CuPtr)
  @assert CUDNN_SOFTMAX_FAST <= algorithm <= CUDNN_SOFTMAX_ACCURATE
  @assert CUDNN_SOFTMAX_MODE_INSTANCE <= mode <= CUDNN_SOFTMAX_MODE_CHANNEL
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @cudnncall(:cudnnSoftmaxForward, (Handle, Cint, Cint, Ptr{Void}, Tensor4dDescriptor, Ptr{Void},
                                    Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, algorithm, mode, alpha_ptr, src_desc, src.p, beta_ptr, dest_desc, dest.p)
end

function softmax_backward{T<:AbstractFloat}(handle::Handle, algorithm::Int, mode::Int,
    alpha::T, src_desc::Tensor4dDescriptor, src::CuPtr, srcdiff_desc::Tensor4dDescriptor, srcdiff::CuPtr,
    beta::T, destdiff_desc::Tensor4dDescriptor, destdiff::CuPtr)
  @assert CUDNN_SOFTMAX_FAST <= algorithm <= CUDNN_SOFTMAX_ACCURATE
  @assert CUDNN_SOFTMAX_MODE_INSTANCE <= mode <= CUDNN_SOFTMAX_MODE_CHANNEL
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @cudnncall(:cudnnSoftmaxBackward, (Handle, Cint, Cint, Ptr{Void}, Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void}, Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, algorithm, mode, alpha_ptr, src_desc, src.p, srcdiff_desc, srcdiff.p,
             beta_ptr, destdiff_desc, destdiff.p)
end

const CUDNN_POOLING_MAX     = 0
const CUDNN_POOLING_AVERAGE = 1

function create_pooling_descriptor()
  desc = PoolingDescriptor[0]
  @cudnncall(:cudnnCreatePoolingDescriptor, (Ptr{PoolingDescriptor},), desc)
  return desc[1]
end
function set_pooling_descriptor(desc::PoolingDescriptor, mode::Int, dims::NTuple{2, Int}, stride::NTuple{2, Int}, padding::NTuple{2, Int})
  @assert CUDNN_POOLING_MAX <= mode <= CUDNN_POOLING_AVERAGE
  w,h = dims
  pad_w, pad_h = padding
  stride_w, stride_h = stride
  @cudnncall(:cudnnSetPooling2dDescriptor, (PoolingDescriptor, Cint, Cint, Cint,Cint, Cint,Cint, Cint,Cint),
             desc, mode, CUDNN_NOT_PROPAGATE_NAN, h,w, pad_w, pad_h, stride_h, stride_w)
end

function create_pooling_descriptor(mode::Int, dims::NTuple{2,Int}, stride::NTuple{2,Int}, padding::NTuple{2,Int})
  desc = create_pooling_descriptor()
  set_pooling_descriptor(desc, mode, dims, stride, padding)
  return desc
end

function destroy_pooling_descriotpr(desc::PoolingDescriptor)
  @cudnncall(:cudnnDestroyPoolingDescriptor, (PoolingDescriptor,), desc)
end

function pooling_forward{T<:AbstractFloat}(handle::Handle, pooling::PoolingDescriptor, alpha::T,
                                           src_desc::Tensor4dDescriptor, src::CuPtr, beta::T,
                                           dest_desc::Tensor4dDescriptor, dest::CuPtr)
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @cudnncall(:cudnnPoolingForward, (Handle, PoolingDescriptor, Ptr{Void},
                                    Tensor4dDescriptor, Ptr{Void}, Ptr{Void},
                                    Tensor4dDescriptor, Ptr{Void}),
             handle, pooling, alpha_ptr,
             src_desc, src.p, beta_ptr,
             dest_desc, dest.p)
end
function pooling_backward{T<:AbstractFloat}(handle::Handle, pooling::PoolingDescriptor, alpha::T,
    src_desc::Tensor4dDescriptor, src::CuPtr, srcdiff_desc::Tensor4dDescriptor, srcdiff::CuPtr,
    dest_desc::Tensor4dDescriptor, dest::CuPtr, beta::T, destdiff_desc::Tensor4dDescriptor, destdiff::CuPtr)
  alpha_ptr = T[alpha]
  beta_ptr = T[beta]
  @cudnncall(:cudnnPoolingBackward, (Handle, PoolingDescriptor, Ptr{Void}, Tensor4dDescriptor, Ptr{Void},
                                     Tensor4dDescriptor, Ptr{Void},Tensor4dDescriptor, Ptr{Void},
                                     Ptr{Void}, Tensor4dDescriptor, Ptr{Void}),
             handle, pooling, alpha_ptr, src_desc, src.p, srcdiff_desc, srcdiff.p,
             dest_desc, dest.p, beta_ptr, destdiff_desc, destdiff.p)
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
