export CuVec
module CuVec
using ..Mocha

function cuda_geometry(sp_dim::Int, chann::Int, num::Int)
  x_block = int(ceil(float64(sp_dim)/CUDA.THREADS_PER_BLOCK_X))
  y_block = int(ceil(float64(chann)/CUDA.THREADS_PER_BLOCK_Y))
  z_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_Z))
  return ((x_block,y_block,z_block),
          (CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z))
end

for (ctype, dtype) in [(:float, Float32), (:double, Float64)]
  # define add!, sub!, mul!, div!, div2!
  for name in [:add, :sub, :mul, :div, :div2]
    @eval begin
      function $(symbol("$(name)!"))(sys::System{CuDNNBackend}, ::Type{$dtype}, X, Y,
          spatial_dim::Int, channels::Int, num::Int)
        X = convert(Ptr{Void},X)
        Y = convert(Ptr{Void},Y)
        cuda_dim = cuda_geometry(spatial_dim, channels, num)
        kernel = sys.backend.mocha.$(symbol("elem_$(name)_$ctype"))
        CUDA.launch(kernel, cuda_dim..., (X, Y, spatial_dim, channels, num))
      end
    end
  end

  # define add_scal!
  @eval begin
    function add_scal!(sys::System{CuDNNBackend}, ::Type{$dtype}, X, Y,
        spatial_dim::Int, channels::Int, num::Int)
      X = convert(Ptr{Void}, X)
      Y = convert($dtype, Y)
      cuda_dim = cuda_geometry(spatial_dim, channels, num)
      kernel = sys.backend.mocha.$(symbol("add_scal_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,spatial_dim,channels,num))
    end
  end

  # define mul_scal!
  @eval begin
    function mul_scal!(sys::System{CuDNNBackend}, ::Type{$dtype}, X, Y,
        spatial_dim::Int, channels::Int, num::Int)
      X = convert(Ptr{Void}, X)
      Y = convert($dtype, Y)
      cuda_dim = cuda_geometry(spatial_dim, channels, num)
      kernel = sys.backend.mocha.$(symbol("mul_scal_$ctype"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,spatial_dim,channels,num))
    end
  end
end

# define add!, sub!, mul!, div!, div2! for blobs
for name in [:add, :sub, :mul, :div, :div2]
  @eval begin
    function $(symbol("$(name)!")){T}(sys::System{CuDNNBackend}, X::CuTensorBlob{T}, Y::CuTensorBlob{T})
      width, height, channels, num = size(X)
      sp_dim = width*height
      $(symbol("$(name)!"))(sys, T, X.ptr.p, Y.ptr.p, sp_dim, channels, num)
    end
  end
end
function add_scal!{T}(sys::System{CuDNNBackend}, X::CuTensorBlob{T}, Y)
  Y = convert(T, Y)
  width, height, channels, num = size(X)
  sp_dim = width*height
  add_scal!(sys, T, X.ptr.p, Y, sp_dim, channels, num)
end
function mul_scal!{T}(sys::System{CuDNNBackend}, X::CuTensorBlob{T}, Y)
  Y = convert(T, Y)
  width, height, channels, num = size(X)
  sp_dim = width*height
  mul_scal!(sys, T, X.ptr.p, Y, sp_dim, channels, num)
end

for (postfix, dt1, dt2) in [(:fi, Float32, Int), (:di, Float64, Int),
                            (:ff, Float32, Float32), (:dd, Float64, Float64)]
  @eval begin
    function pow!(sys::System{CuDNNBackend}, ::Type{$dt1}, X, Y::$dt2,
        spatial_dim::Int, channels::Int, num::Int)
      X = convert(Ptr{Void}, X)
      cuda_dim = cuda_geometry(spatial_dim, channels, num)
      kernel = sys.backend.mocha.$(symbol("elem_pow_$postfix"))
      CUDA.launch(kernel, cuda_dim..., (X,Y,spatial_dim,channels,num))
    end
  end
end

end # module cuVec
