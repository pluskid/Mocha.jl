export CuBLAS

module CuBLAS
using ..CUDA

# cublasStatus_t
const CUBLAS_STATUS_SUCCESS         = 0
const CUBLAS_STATUS_NOT_INITIALIZED = 1
const CUBLAS_STATUS_ALLOC_FAILED    = 3
const CUBLAS_STATUS_INVALID_VALUE   = 7
const CUBLAS_STATUS_ARCH_MISMATCH   = 8
const CUBLAS_STATUS_MAPPING_ERROR   = 11
const CUBLAS_STATUS_EXECUTION_FAILED= 13
const CUBLAS_STATUS_INTERNAL_ERROR  = 14
const CUBLAS_STATUS_NOT_SUPPORTED   = 15
const CUBLAS_STATUS_LICENSE_ERROR   = 16

immutable CuBLASError <: Exception
  code :: Int
end
const cublas_error_description = [
  CUBLAS_STATUS_SUCCESS         => "Success",
  CUBLAS_STATUS_NOT_INITIALIZED => "Not initialized",
  CUBLAS_STATUS_ALLOC_FAILED    => "Alloc failed",
  CUBLAS_STATUS_INVALID_VALUE   => "Invalid value",
  CUBLAS_STATUS_ARCH_MISMATCH   => "Arch mismatch",
  CUBLAS_STATUS_MAPPING_ERROR   => "Mapping error",
  CUBLAS_STATUS_EXECUTION_FAILED=> "Execution failed",
  CUBLAS_STATUS_INTERNAL_ERROR  => "Internal error",
  CUBLAS_STATUS_NOT_SUPPORTED   => "Not supported",
  CUBLAS_STATUS_LICENSE_ERROR   => "License error"
]
import Base.show
show(io::IO, error::CuBLASError) = print(io, cublas_error_description[error.code])

macro cublascall(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), "libcublas"), Cint, $argtypes, $(args...)  )
    if int(_curet) != CUBLAS_STATUS_SUCCESS
      throw(CuBLASError(int(_curet)))
    end
  end
end

typealias Handle Ptr{Void}
typealias StreamHandle Ptr{Void}

function create()
  handle = Handle[0]
  @cublascall(:cublasCreate_v2, (Ptr{Handle},), handle)
  return handle[1]
end
function destroy(handle :: Handle)
  @cublascall(:cublasDestroy_v2, (Handle,), handle)
end
function set_stream(handle::Handle, stream::StreamHandle)
  @cublascall(:cublasSetStream_v2, (Handle, StreamHandle), handle, stream)
end
function get_stream(handle::Handle)
  s_handle = StreamHandle[0]
  @cublascall(:cublasGetStream_v2, (Handle, Ptr{StreamHandle}), handle, s_handle)
  return s_handle[1]
end


############################################################
# Copy a vector from host to device
############################################################
function set_vector(n::Int, elem_size::Int, src::Ptr{Void}, incx::Int, dest::Ptr{Void}, incy::Int)
  @cublascall(:cublasSetVector, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint),
      n, elem_size, src, incx, dest, incy)
end
function set_vector(n::Int, elem_size::Int, src::Ptr{Void}, incx::Int, dest::CuPtr, incy::Int)
  set_vector(n, elem_size, src, incx, convert(Ptr{Void}, dest.p), incy)
end
function set_vector{T}(src::Array{T}, incx::Int, dest::CuPtr, incy::Int)
  elem_size = sizeof(T)
  n = length(src)
  src_buf = convert(Ptr{Void}, src)
  set_vector(n, elem_size, src_buf, incx, dest, incy)
end
set_vector{T}(src::Array{T}, dest::CuPtr) = set_vector(src, 1, dest, 1)

############################################################
# Copy a vector from device to host
############################################################
function get_vector(n::Int, elem_size::Int, src::CuPtr, incx::Int, dest::Ptr{Void}, incy::Int)
  @cublascall(:cublasGetVector, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint),
      n, elem_size, src.p, incx, dest, incy)
end
function get_vector{T}(src::CuPtr, incx::Int, dest::Array{T}, incy::Int)
  elem_size = sizeof(T)
  n = length(dest)
  dest_buf = convert(Ptr{Void}, dest)
  get_vector(n, elem_size, src, incx, dest_buf, incy)
end
get_vector{T}(src::CuPtr, dest::Array{T}) = get_vector(src, 1, dest, 1)


############################################################
# y = α y
############################################################
function scal(handle::Handle, n::Int, alpha::Float32, x::CuPtr, incx::Int)
  alpha_box = Float32[alpha]
  @cublascall(:cublasSscal_v2, (Handle, Cint, Ptr{Void}, Ptr{Void}, Cint),
              handle, n, alpha_box, x.p, incx)
end
function scal(handle::Handle, n::Int, alpha::Float64, x::CuPtr, incx::Int)
  alpha_box = Float64[alpha]
  @cublascall(:cublasSscal_v2, (Handle, Cint, Ptr{Void}, Ptr{Void}, Cint),
              handle, n, alpha_box, x.p, incx)
end

############################################################
# y = α x + y
############################################################
function axpy(handle::Handle, n::Int, alpha::Float32, x::CuPtr, incx::Int, y::CuPtr, incy::Int)
  alpha_box = Float32[alpha]
  @cublascall(:cublasSaxpy_v2, (Handle, Cint, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Cint),
      handle, n, alpha_box, x.p, incx, y.p, incy)
end
function axpy(handle::Handle, n::Int, alpha::Float64, x::CuPtr, incx::Int, y::CuPtr, incy::Int)
  alpha_box = Float64[alpha]
  @cublascall(:cublasDaxpy_v2, (Handle, Cint, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Cint),
      handle, n, alpha_box, x.p, incx, y.p, incy)
end

############################################################
# vector dot product
############################################################
function dot(handle::Handle, n::Int, x::CuPtr, incx::Int, y::CuPtr, incy::Int)
  result = Float32[0]
  @cublascall(:cublasSdot_v2, (Handle, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, Ptr{Void}),
      handle, n, x.p, incx, y.p, incy, result)
  return result[1]
end
function dot(handle::Handle, n::Int, x::CuPtr, incx::Int, y::CuPtr, incy::Int)
  result = Float64[0]
  @cublascall(:cublasDdot_v2, (Handle, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint, Ptr{Void}),
      handle, n, x.p, incx, y.p, incy, result)
  return result[1]
end

############################################################
# cublasOperation_t
############################################################
const OP_N=0
const OP_T=1
const OP_C=2

############################################################
# C = α A * B + β C
############################################################
function gemm{T}(handle::Handle, trans_a::Int, trans_b::Int, m::Int, n::Int, k::Int,
    alpha::T, A::CuPtr, lda::Int, B::CuPtr, ldb::Int, beta::T, C::CuPtr, ldc::Int)
  @assert OP_N <= trans_a <= OP_C
  @assert OP_N <= trans_b <+ OP_C
  alpha_box = T[alpha]
  beta_box = T[beta]
  gemm_impl(handle, trans_a, trans_b, m, n, k, alpha_box, A, lda, B, ldb, beta_box, C, ldc)
end

function gemm_impl(handle::Handle, trans_a::Int, trans_b::Int, m::Int, n::Int, k::Int,
    alpha_box::Array{Float32}, A::CuPtr, lda::Int, B::CuPtr, ldb::Int, beta_box::Array{Float32}, C::CuPtr, ldc::Int)
  @cublascall(:cublasSgemm_v2, (Handle, Cint,Cint, Cint,Cint,Cint, Ptr{Void},
      Ptr{Void},Cint, Ptr{Void},Cint, Ptr{Void}, Ptr{Void},Cint),
      handle, trans_a, trans_b, m, n, k, alpha_box, A.p, lda, B.p, ldb, beta_box, C.p, ldc)
end
function gemm_impl(handle::Handle, trans_a::Int, trans_b::Int, m::Int, n::Int, k::Int,
    alpha_box::Array{Float64}, A::CuPtr, lda::Int, B::CuPtr, ldb::Int, beta_box::Array{Float64}, C::CuPtr, ldc::Int)
  @cublascall(:cublasDgemm_v2, (Handle, Cint,Cint, Cint,Cint,Cint, Ptr{Void},
      Ptr{Void},Cint, Ptr{Void},Cint, Ptr{Void}, Ptr{Void},Cint),
      handle, trans_a, trans_b, m, n, k, alpha_box, A.p, lda, B.p, ldb, beta_box, C.p, ldc)
end

end # module
