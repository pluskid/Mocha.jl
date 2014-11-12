function im2col{T}(img::Array{T}, n::Int, col::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  im2col_impl(img[:,:,:,n], col, width, height, channels, kernel, pad, stride)
end

function im2col_impl{T}(img::Array{T}, col::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

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

function col2im{T}(col::Array{T}, img::Array{T}, n::Int, img_buf::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  col2im_impl(col, img_buf, width, height, channels, kernel, pad, stride)
  img[:,:,:,n] = img_buf
end

function col2im_impl{T}(col::Array{T}, img::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
  channels_col = channels * kernel_h * kernel_w

  fill!(img, 0)
  for c = 0:channels_col-1
    w_offset = c % kernel_w
    h_offset = div(c, kernel_w) % kernel_h
    c_im = div(c, kernel_w * kernel_h)
    for h = 0:height_col-1
      for w = 0:width_col-1
        h_pad = h * stride_h - pad_h + h_offset
        w_pad = w * stride_w - pad_w + w_offset
        if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
          @inbounds img[1 + (c_im * height + h_pad) * width + w_pad] +=
              col[1 + (c * height_col + h) * width_col + w]
        end
      end
    end
  end
end

