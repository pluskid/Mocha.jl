const IM2COL_FLOAT_HANDLE = Libdl.dlsym(Native.library, :im2col_float)
const IM2COL_DOUBLE_HANDLE = Libdl.dlsym(Native.library, :im2col_double)
const COL2IM_FLOAT_HANDLE = Libdl.dlsym(Native.library, :col2im_float)
const COL2IM_DOUBLE_HANDLE = Libdl.dlsym(Native.library, :col2im_double)

function im2col{T}(img::Array{T}, n::Int, col::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  img_ptr = pointer(img) + (width*height*channels) * (n-1) * sizeof(T)
  col_ptr = pointer(col)
  im2col_impl(img_ptr, col_ptr, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end

function im2col_impl(img::Ptr{Float32}, col::Ptr{Float32}, width::Int, height::Int, channels::Int,
    kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  ccall(IM2COL_FLOAT_HANDLE, Void, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint,
        Cint, Cint, # kernel
        Cint, Cint, # pad
        Cint, Cint, # stride
      ), img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end
function im2col_impl(img::Ptr{Float64}, col::Ptr{Float64}, width::Int, height::Int, channels::Int,
    kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  ccall(IM2COL_DOUBLE_HANDLE, Void, (Ptr{Float64}, Ptr{Float64}, Cint, Cint, Cint,
        Cint, Cint, # kernel
        Cint, Cint, # pad
        Cint, Cint, # stride
      ), img, col, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end


function col2im{T}(col::Array{T}, img::Array{T}, n::Int, img_buf::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  # ignore img_buf
  img_ptr = pointer(img) + (width*height*channels) * (n-1) * sizeof(T)
  col_ptr = pointer(col)
  col2im_impl(col_ptr, img_ptr, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end

function col2im_impl(col::Ptr{Float32}, img::Ptr{Float32}, width::Int, height::Int, channels::Int,
    kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  ccall(COL2IM_FLOAT_HANDLE, Void, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint,
        Cint, Cint, # kernel
        Cint, Cint, # pad
        Cint, Cint, # stride
      ), col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end
function col2im_impl(col::Ptr{Float64}, img::Ptr{Float64}, width::Int, height::Int, channels::Int,
    kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  ccall(COL2IM_DOUBLE_HANDLE, Void, (Ptr{Float64}, Ptr{Float64}, Cint, Cint, Cint,
        Cint, Cint, # kernel
        Cint, Cint, # pad
        Cint, Cint, # stride
      ), col, img, width, height, channels, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h)
end
