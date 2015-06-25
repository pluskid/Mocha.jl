const MAX_POOL_FWD_FLOAT = Libdl.dlsym(Native.library, :max_pooling_fwd_float)
const MAX_POOL_FWD_DOUBLE = Libdl.dlsym(Native.library, :max_pooling_fwd_double)
const MAX_POOL_BWD_FLOAT = Libdl.dlsym(Native.library, :max_pooling_bwd_float)
const MAX_POOL_BWD_DOUBLE = Libdl.dlsym(Native.library, :max_pooling_bwd_double)

const MEAN_POOL_FWD_FLOAT = Libdl.dlsym(Native.library, :mean_pooling_fwd_float)
const MEAN_POOL_FWD_DOUBLE = Libdl.dlsym(Native.library, :mean_pooling_fwd_double)
const MEAN_POOL_BWD_FLOAT = Libdl.dlsym(Native.library, :mean_pooling_bwd_float)
const MEAN_POOL_BWD_DOUBLE = Libdl.dlsym(Native.library, :mean_pooling_bwd_double)

function max_pooling_forward(input::Array{Float32}, output::Array{Float32}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  ccall(MAX_POOL_FWD_FLOAT, Void,
      (Ptr{Float32}, Ptr{Float32}, Ptr{Csize_t},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output, mask,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end
function max_pooling_forward(input::Array{Float64}, output::Array{Float64}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  ccall(MAX_POOL_FWD_DOUBLE, Void,
      (Ptr{Float64}, Ptr{Float64}, Ptr{Csize_t},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output, mask,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end

function max_pooling_backward(input::Array{Float32}, output::Array{Float32}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  ccall(MAX_POOL_BWD_FLOAT, Void,
      (Ptr{Float32}, Ptr{Float32}, Ptr{Csize_t},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output, mask,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end
function max_pooling_backward(input::Array{Float64}, output::Array{Float64}, mask::Array{Csize_t}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)
  ccall(MAX_POOL_BWD_DOUBLE, Void,
      (Ptr{Float64}, Ptr{Float64}, Ptr{Csize_t},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output, mask,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end

function mean_pooling_forward(input::Array{Float32}, output::Array{Float32}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  ccall(MEAN_POOL_FWD_FLOAT, Void,
      (Ptr{Float32}, Ptr{Float32},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end
function mean_pooling_forward(input::Array{Float64}, output::Array{Float64}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  ccall(MEAN_POOL_FWD_DOUBLE, Void,
      (Ptr{Float64}, Ptr{Float64},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end

function mean_pooling_backward(input::Array{Float32}, output::Array{Float32}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  ccall(MEAN_POOL_BWD_FLOAT, Void,
      (Ptr{Float32}, Ptr{Float32},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end
function mean_pooling_backward(input::Array{Float64}, output::Array{Float64}, layer)
  width, height, channels, num = size(input)
  pooled_width = size(output, 1)
  pooled_height = size(output, 2)

  ccall(MEAN_POOL_BWD_DOUBLE, Void,
      (Ptr{Float64}, Ptr{Float64},
      Cint, Cint, Cint, Cint,
      Cint, Cint,
      Cint, Cint, Cint, Cint, Cint, Cint),
      input, output,
      width, height, channels, num,
      pooled_width, pooled_height,
      layer.kernel[1], layer.kernel[2],
      layer.pad[1], layer.pad[2], layer.stride[1], layer.stride[2])
end

