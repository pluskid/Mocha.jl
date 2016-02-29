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
          convert(data_type,state.layer.scale), get_ptr(output), 1)
    end

    if state.layer.shift != 0
      # output += shift
      CuVec.add_scal!(backend, data_type, get_ptr(output).p, convert(data_type, state.layer.shift), len)
    end

    # output = output ^ power
    if state.layer.power != 1
      if state.layer.power == 2
        CuVec.mul!(backend, data_type, get_ptr(output).p, get_ptr(output).p, len)
      else
        CuVec.pow!(backend, data_type, get_ptr(output).p,
            isinteger(state.layer.power) ? round(Int, state.layer.power) : convert(data_type, state.layer.power),
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
    diff = diffs[i]
    if isa(diff, NullBlob)
      continue
    end

    len = length(inputs[i])
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
            get_ptr(input), 1, get_ptr(diff), 1)
        if state.layer.shift != 0
          CuVec.add_scal!(backend, data_type, get_ptr(diff).p, pow_scale * state.layer.shift, len)
        end
      elseif state.layer.shift == 0
        # dO/dI = power * scale * (scale * I) ^ (power - 1)
        #       = power * O / I
        CuBLAS.axpy(backend.cublas_ctx, length(input), convert(data_type,state.layer.power),
            get_ptr(output), 1, get_ptr(diff), 1)
        CuVec.div!(backend, data_type, get_ptr(diff).p, get_ptr(input).p, len)
      else
        # general case
        # dO/dI = power * scale * (scale * I + shift) ^ (power - 1)
        #       = power * scale * O / (scale * I + shift)
        copy!(diff, input)
        if state.layer.scale != 1
          CuBLAS.scal(backend.cublas_ctx, length(diff),
              convert(data_type,state.layer.scale), get_ptr(diff), 1)
        end
        CuVec.add_scal!(backend, data_type, get_ptr(diff).p, state.layer.shift, len)
        CuVec.div2!(backend, data_type, get_ptr(output).p, get_ptr(diff).p, len)
        CuBLAS.scal(backend.cublas_ctx, length(diff), pow_scale, get_ptr(diff), 1)
      end
    end
    CuVec.mul!(backend, data_type, get_ptr(diff).p, get_ptr(state.blobs_diff[i]).p, len)
  end
end
