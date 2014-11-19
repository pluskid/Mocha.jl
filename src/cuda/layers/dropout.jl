function setup_etc(sys::System{CuDNNBackend}, layer::DropoutLayer, inputs::Vector{Blob})
  cuda_rand_states = Array(CuPtr, length(inputs))
  kernel = sys.backend.mocha.dropout_init
  rnd_state_size = Csize_t[0]
  CUDA.launch(sys.backend.mocha.dropout_alloc_size, 1, 1, (rnd_state_size, ))
  rnd_state_size = rnd_state_size[1]
  @debug("rnd_state_size = $rnd_state_size")

  for i = 1:length(inputs)
    len = length(inputs[i])
    cuda_rand_states[i] = CUDA.cualloc(Uint8, rnd_state_size*len)
    x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
    CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (cuda_rand_states[i], len))
  end

  return cuda_rand_states
end

function destroy_etc(sys::System{CuDNNBackend}, state::DropoutLayerState)
  for i = 1:length(state.etc)
    CUDA.free(state.etc[i])
  end
end

function forward(sys::System{CuDNNBackend}, state::DropoutLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    len = length(inputs[i])
    x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
    data_type = eltype(inputs[i])
    if data_type == Float32
      kernel = sys.backend.mocha.dropout_forward_float
    elseif data_type == Float64
      kernel = sys.backend.mocha.dropout_forward_double
    end

    CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
        (state.etc[i], length(inputs[i]), inputs[i].ptr.p,
        state.blobs[i].ptr.p, state.rand_vals[i].ptr.p,
        state.ratio, state.scale))
  end
end

function backward(sys::System{CuDNNBackend}, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      len = length(inputs[i])
      x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
      data_type = eltype(inputs[i])
      if data_type == Float32
        kernel = sys.backend.mocha.dropout_backward_float
      elseif data_type == Float64
        kernel = sys.backend.mocha.dropout_backward_double
      end

      CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
          (state.etc[i], length(inputs[i]), diffs[i].ptr.p,
          state.blobs_diff[i].ptr.p, state.rand_vals[i].ptr.p,
          state.ratio, state.scale))
    end
  end
end
