export im2col, col2im

function im2col(data_im, data_col, kernel::Vector{Int}, pad::Vector{Int}, stride::Vector{Int})

  height = size(data_im,3)
  width  = size(data_im,4)

  height_col = (height + 2*pad[1] - kernel[1]) / stride[1] + 1
  width_col  = (width + 2*pad[2] - kernel[2]) / stride[2] + 1
  chann_col  = size(data_im,2) * kernel[1] * kernel[2]

  for c = 0:(chann_col-1)
    w_offset = c % kernel[2]
    h_offset = (c / kernel[2]) % kernel[1]
    c_im = c / kernel[1] / kernel[2]

    for h = 0:(height_col-1)
      for w = 0:(width_col-1)
        h_pad = h*stride[1] - pad[1] + h_offset
        w_pad = w*stride[2] - pad[2] + w_offset
        idx = 1 + (c*height_col+h)*width_col+w
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[idx] = data_im[1 + (c_im*height+h_pad)*width+w_pad]
        else
          data_col[idx] = 0
        end
      end
    end
  end
end

function col2im(data_col, data_im, patch::Vector{Int}, pad::Vector{Int}, stride::Vector{Int})

  fill!(data_im.data, 0)

  height = size(data_im,3)
  width  = size(data_im,4)

  height_col = (height + 2*pad[1] - patch[1]) / stride[1] + 1
  width_col  = (width  + 2*pad[2] - patch[2]) / stride[2] + 1
  chann_col = size(data_im,2) * patch[1] * patch[2]

  for c = 0:(chann_col-1)
    w_offset = c % patch[2]
    h_offset = (c / patch[2]) % patch[1]
    c_im = c / patch[1] / patch[2]

    for h = 0:(height_col-1)
      for w = 0:(width_col-1)
        h_pad = h * stride[1] - pad[1] + h_offset
        w_pad = w * stride[2] - pad[2] + w_offset
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[1 + (c_im*height+h_pad) * width + w_pad] +=
              data_col[1 + (c*height_col+h) * width_col + w]
        end
      end
    end
  end
end
