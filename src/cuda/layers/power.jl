############################################################
# Power Layer
############################################################

function forward(backend::GPUBackend, state::PowerLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    len = length(input)
    data_type = eltype(input)

    # output = input
    copy!(output, input)

    # output *= scale
    if state.layer.scale != 1
      CuBLAS.scal(backend.cublas_ctx, length(output),
          convert(data_type,state.layer.scale), output.ptr, 1)
    end

    if state.layer.shift != 0
      # output += shift
      CuVec.add_scal!(backend, data_type, output.ptr.p, convert(data_type, state.layer.shift), len)
    end

    # output = output ^ power
    if state.layer.power != 1
      if state.layer.power == 2
        CuVec.mul!(backend, data_type, output.ptr.p, output.ptr.p, len)
      else
        CuVec.pow!(backend, data_type, output.ptr.p,
            isinteger(state.layer.power) ? round(Int64, state.layer.power) : convert(data_type, state.layer.power),
            len)
      end
    end
  end
end

function backward(backend::GPUBackend, state::PowerLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  data_type = eltype(inputs[1])
  pow_scale = convert(data_type,state.layer.power * state.layer.scale)
  for i = 1:length(inputs)
    len = length(inputs[i])

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
        CuBLAS.axpy(backend.cublas_ctx, length(input), convert(data_type, pow_scale*state.layer.scale),
            input.ptr, 1, diff.ptr, 1)
        if state.layer.shift != 0
          CuVec.add_scal!(backend, data_type, diff.ptr.p, pow_scale * state.layer.shift, len)
        end
      elseif state.layer.shift == 0
        # dO/dI = power * scale * (scale * I) ^ (power - 1)
        #       = power * O / I
        CuBLAS.axpy(backend.cublas_ctx, length(input), convert(data_type,state.layer.power),
            output.ptr, 1, diff.ptr, 1)
        CuVec.div!(backend, data_type, diff.ptr.p, input.ptr.p, len)
      else
        # general case
        # dO/dI = power * scale * (scale * I + shift) ^ (power - 1)
        #       = power * scale * O / (scale * I + shift)
        copy!(diff, input)
        if state.layer.scale != 1
          CuBLAS.scal(backend.cublas_ctx, length(diff),
              convert(data_type,state.layer.scale), diff.ptr, 1)
        end
        CuVec.add_scal!(backend, data_type, diff.ptr.p, state.layer.shift, len)
        CuVec.div2!(backend, data_type, output.ptr.p, diff.ptr.p, len)
        CuBLAS.scal(backend.cublas_ctx, length(diff), pow_scale, diff.ptr, 1)
      end
    end
    CuVec.mul!(backend, data_type, diff.ptr.p, state.blobs_diff[i].ptr.p, len)
  end
end
