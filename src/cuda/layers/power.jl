############################################################
# Power Layer
############################################################

function forward(sys::System{CuDNNBackend}, state::PowerLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    width, height, channels, num = size(input)
    spatial_dim = width*height
    data_type = eltype(input)

    # output = input
    copy!(output, input)

    # output *= scale
    if state.layer.scale != 1
      CuBLAS.scal(sys.backend.cublas_ctx, length(output), 
          convert(data_type,state.layer.scale), output.ptr, 1)
    end

    if state.layer.shift != 0
      # output += shift
      CuVec.add_scal!(sys, data_type, output.ptr.p, convert(data_type, state.layer.shift), 
          spatial_dim, channels, num)
    end

    # output = output ^ power
    if state.layer.power != 1
      if state.layer.power == 2
        CuVec.mul!(sys, data_type, output.ptr.p, output.ptr.p, spatial_dim, channels, num)
      else
        CuVec.pow!(sys, data_type, output.ptr.p, state.layer.power, 
            spatial_dim, channels, num)
      end
    end
  end
end

function backward(sys::System{CuDNNBackend}, state::PowerLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  data_type = eltype(inputs[1])
  pow_scale = convert(data_type,state.layer.power * state.layer.scale)
  for i = 1:length(inputs)
    width, height, channels, num = size(inputs[i])
    spatial_dim = width*height

    diff = diffs[i]
    if state.layer.power == 1 || state.layer.scale == 0
      # trivial case, derivative is constant
      fill!(diff, pow_scale)
    else
      input = inputs[i]
      output = state.blobs[i]

      erase!(diff)

      if state.layer.power == 2
        # dO/dI = 2 * scale * (scale * I + shift)
        #       = pow_scale * scale * I + pow_scale * shift
        CuBLAS.axpy(sys.backend.cublas_ctx, length(input), pow_scale*state.layer.scale,
            input.ptr, 1, diff.ptr, 1)
        if state.layer.shift != 0
          CuVec.add_scal!(sys, data_type, diff.ptr.p, pow_scale * state.layer.shift,
            spatial_dim, channels, num)
        end
      elseif state.layer.shift == 0
        # dO/dI = power * scale * (scale * I) ^ (power - 1)
        #       = power * O / I
        CuBLAS.axpy(sys.backend.cublas_ctx, length(input), convert(data_type,state.layer.power),
            output.ptr, 1, diff.ptr, 1)
        CuVec.div!(sys, data_type, diff.ptr.p, input.ptr.p, spatial_dim, channels, num)
      else
        # general case
        # dO/dI = power * scale * (scale * I + shift) ^ (power - 1)
        #       = power * scale * O / (scale * I + shift)
        copy!(diff, input)
        if state.layer.scale != 1
          CuBLAS.scal(sys.backend.cublas_ctx, length(diff), 
              convert(data_type,state.layer.scale), diff.ptr, 1)
        end
        CuVec.add_scal!(sys, data_type, diff.ptr.p, state.layer.shift, 
            spatial_dim, channels, num)
        CuVec.div2!(sys, data_type, output.ptr.p, diff.ptr.p,
            spatial_dim, channels, num)
        CuBLAS.scal(sys.backend.cublas_ctx, length(diff), pow_scale, diff.ptr, 1)
      end
    end
    CuVec.mul!(sys, data_type, diff.ptr.p, state.blobs_diff[i].ptr.p,
        spatial_dim, channels, num)
  end
end
