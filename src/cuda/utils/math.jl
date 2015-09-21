export CuVec
module CuVec
using ..Mocha

const THREADS_PER_BLOCK = 128
function cuda_geometry(len::Int)
  x_block = round(Int64, ceil(convert(Float64, len)/THREADS_PER_BLOCK))
  return (x_block, THREADS_PER_BLOCK)
end

for (ctype, dtype) in [(:float, Float32), (:double, Float64)]
  # define add!, sub!, mul!, div!, div2!
  for name in [:add, :sub, :mul, :div, :div2]
    @eval begin
      function $(symbol("$(name)!"))(backend::GPUBackend, ::Type{$dtype}, X, Y, len::Int)
        X = convert(Ptr{Void},X)
        Y = convert(Ptr{Void},Y)
        cuda_dim = cuda_geometry(len)
        kernel = backend.mocha.$(symbol("elem_$(name)_$ctype"))
        CUDA.launch(kernel, cuda_dim..., (X, Y, len))
      end
    end
  end

  # define add_scal!
  @eval begin
    function add_scal!(backend::GPUBackend, ::Type{$dtype}, X, Y, len::Int)
      X = convert(Ptr{Void}, X)
      Y = convert($dtype, Y)
      cuda_dim = cuda_geometry(len)
      kernel = backend.mocha.$(symbol("add_scal_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,len))
    end
  end

  # define log!
  @eval begin
    function log!(backend::GPUBackend, ::Type{$dtype}, X, len::Int)
      X = convert(Ptr{Void}, X)
      cuda_dim = cuda_geometry(len)
      kernel = backend.mocha.$(symbol("elem_log_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,len))
    end
  end

  # define exp!
  @eval begin
    function exp!(backend::GPUBackend, ::Type{$dtype}, X, len::Int)
      X = convert(Ptr{Void}, X)
      cuda_dim = cuda_geometry(len)
      kernel = backend.mocha.$(symbol("elem_exp_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,len))
    end
  end

  # define mul_scal!
  @eval begin
    function mul_scal!(backend::GPUBackend, ::Type{$dtype}, X, Y, len::Int)
      X = convert(Ptr{Void}, X)
      Y = convert($dtype, Y)
      cuda_dim = cuda_geometry(len)
      kernel = backend.mocha.$(symbol("mul_scal_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,len))
    end
  end
end

# define add!, sub!, mul!, div!, div2! for blobs
for name in [:add, :sub, :mul, :div, :div2]
  @eval begin
    function $(symbol("$(name)!")){T}(backend::GPUBackend, X::CuTensorBlob{T}, Y::CuTensorBlob{T})
      len = length(X)
      $(symbol("$(name)!"))(backend, T, X.ptr.p, Y.ptr.p, len)
    end
  end
end
function add_scal!{T}(backend::GPUBackend, X::CuTensorBlob{T}, Y)
  Y = convert(T, Y)
  len = length(X)
  add_scal!(backend, T, X.ptr.p, Y, len)
end
function mul_scal!{T}(backend::GPUBackend, X::CuTensorBlob{T}, Y)
  Y = convert(T, Y)
  len = length(X)
  mul_scal!(backend, T, X.ptr.p, Y, len)
end
function log!{T}(backend::GPUBackend, X::CuTensorBlob{T})
  log!(backend, T, X.ptr.p, length(X))
end
function exp!{T}(backend::GPUBackend, X::CuTensorBlob{T})
  exp!(backend, T, X.ptr.p, length(X))
end

for (postfix, dt1, dt2) in [(:fi, Float32, Int), (:di, Float64, Int),
                            (:ff, Float32, Float32), (:dd, Float64, Float64)]
  @eval begin
    function pow!(backend::GPUBackend, ::Type{$dt1}, X, Y::$dt2, len::Int)
      X = convert(Ptr{Void}, X)
      cuda_dim = cuda_geometry(len)
      kernel = backend.mocha.$(symbol("elem_pow_$postfix"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,len))
    end
    function pow!{T}(backend::GPUBackend, X::CuTensorBlob{T}, Y::$dt2)
      pow!(backend, T, X.ptr.p, Y, length(X))
    end
  end
end


end # module cuVec
