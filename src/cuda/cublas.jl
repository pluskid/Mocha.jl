export CuBLAS

module CuBLAS
using CUDA

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

macro cublascall(f, argtypes, args...)
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

function set_vector(n::Int, elem_size::Int, src::Ptr{Void}, incx::Int, dest::CuPtr, incy::Int)
  @cublascall(:cublasSetVector, (Cint, Cint, Ptr{Void}, Cint, Ptr{Void}, Cint),
      n, elem_size, src, incx, dest.p, incy)
end
function set_vector{T}(src::Array{T}, incx::Int, dest::CuPtr, incy::Int)
  elem_size = sizeof(T)
  n = length(src)
  src_buf = convert(Ptr{Void}, src)
  set_vector(n, elem_size, src_buf, incx, dest, incy)
end
set_vector{T}(src::Array{T}, dest::CuPtr) = set_vector(src, 1, dest, 1)

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

# cublasOperation_t
const CUBLAS_OP_N=0
const CUBLAS_OP_T=1
const CUBLAS_OP_C=2

# C = α A * B + β C
for dtype in (Float32, Float64)
  if dtype == Float32
    cublas_gemm_func = :cublasSgemm_v2
  elseif dtype == Float64
    cublas_gemm_func = :cublasDgemm_v2
  end
  @eval begin
    function gemm(handle::Handle, trans_a::Int, trans_b::Int, m::Int, n::Int, k::Int,
        alpha::$dtype, A::CuPtr, lda::Int, B::CuPtr, ldb::Int, beta::$dtype, C::CuPtr, ldc::Int)
      @assert CUBLAS_OP_N <= trans_a <= CUBLAS_OP_C
      @assert CUBLAS_OP_N <= trans_b <+ CUBLAS_OP_C
      alpha_box = $dtype[alpha]
      beta_box = $dtype[beta]
      @cublascall($cublas_gemm_func, (Handle, Cint,Cint, Cint,Cint,Cint, Ptr{Void},
          Ptr{Void},Cint, Ptr{Void},Cint, Ptr{Void}, Ptr{Void},Cint),
          handle, trans_a, trans_b, m, n, k, alpha_box, A, lda, B, ldb, beta_box, C, ldc)
    end
  end
end

end # module
