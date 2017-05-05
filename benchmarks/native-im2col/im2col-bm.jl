using Benchmark

################################################################################
# im2col is a bottleneck according to profiling. In this benchmark we try to
# compare the performance of C vs. julia in im2col. It turns out that C is a
# little bit faster, plus creating copy of sub-array could be avoided when
# processing im2col for each image in a mini-batch.
#
# | Row | Function    | Average     | Relative | Replications |
# |-----|-------------|-------------|----------|--------------|
# | 1   | "im2col_jl" | 0.000590041 | 1.77187  | 50           |
# | 2   | "im2col_c"  | 0.000333004 | 1.0      | 50           |
#
# Note if we add omp parallel for to the outer-most for loop, the performance
# deteriorate significantly.
# | Row | Function    | Average     | Relative | Replications |
# |-----|-------------|-------------|----------|--------------|
# | 1   | "im2col_jl" | 0.000831314 | 1.0      | 50           |
# | 2   | "im2col_c"  | 0.00514862  | 6.19335  | 50           |
################################################################################

function im2col{T}(img::Array{T}, col::Array{T}, width::Int, height::Int, channels::Int, kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})
  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
  channels_col = channels * kernel_h * kernel_w

  for c = 0:channels_col-1
    w_offset = c % kernel_w
    h_offset = div(c, kernel_w) % kernel_h
    c_im = div(c, kernel_h * kernel_w) # channel
    for h = 0:height_col-1
      for w = 0:width_col-1
        h_pad = h*stride_h - pad_h + h_offset
        w_pad = w*stride_w - pad_w + w_offset
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            @inbounds col[1 + (c*height_col+h) * width_col + w] =
                img[1 + (c_im * height + h_pad) * width + w_pad]
        else
          @inbounds col[1 + (c*height_col+h) * width_col + w] = 0
        end
      end
    end
  end
end

library = dlopen("./libextim2col.so")
func_handle = dlsym(library, :im2col)
function im2col_native(img::Array{Float64}, col::Array{Float64}, width::Int, height::Int, channels::Int, kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})
  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  ccall(func_handle, Void,
      (Ptr{Float64},Ptr{Float64}, Cint, Cint, Cint,
      Cint, Cint, # kernel
      Cint, Cint, # pad
      Cint, Cint, # stride
      ), img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end


############################################################
# Benchmark data preparation
############################################################
width, height, channels = (28, 28, 50)
kernel = (5,5)
pad = (2,2)
stride = (2,2)
img = rand(width, height, channels)

width_out  = div(width  + 2*pad[1]-kernel[1], stride[1]) + 1
height_out = div(height + 2*pad[2]-kernel[2], stride[2]) + 1
col_buffer = Array{Float64}(width_out, height_out, channels*prod(kernel))
col_buffer2 = zeros(size(col_buffer))

im2col_jl() = im2col(img, col_buffer, width, height, channels, kernel, pad, stride)
im2col_c() = im2col_native(img, col_buffer2, width, height, channels, kernel, pad, stride)

im2col_jl()
im2col_c()

@assert all(abs.(col_buffer-col_buffer2) .< 1e-10)

df = compare([im2col_jl, im2col_c], 50)
println("$df")
