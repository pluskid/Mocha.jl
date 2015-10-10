function setup_etc(backend::GPUBackend, layer::DropoutLayer, inputs::Vector{Blob})
  cuda_rand_states = CuPtr
  kernel = backend.mocha.dropout_init
  rnd_state_size_blob = make_blob(backend, Float64, 1, 1, 1, 1)
  CUDA.launch(backend.mocha.dropout_alloc_size, 1, 1, (rnd_state_size_blob.ptr.p, ))
  rnd_state_size = Float64[0]
  copy!(rnd_state_size, rnd_state_size_blob)
  destroy(rnd_state_size_blob)
  rnd_state_size = round(Int, rnd_state_size[1])

  len = length(inputs[1])
  cuda_rand_states = CUDA.cualloc(UInt8, rnd_state_size*len)
  x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (cuda_rand_states, len))

  # hold copy of input blob, we will restore the inputs after backward computing
  # this is because when used with the cuDNN pooling layer, the in-place modification
  # of the pooling layer output will cause error in computed gradient due to unknown
  # implementation of the pooling backward. So as a workaround, we restore the input
  # after finishing dropout backward computing. See #96 on github for more details.
  input_copy = make_blob(backend, eltype(inputs[1]), size(inputs[1]))

  return (cuda_rand_states, input_copy)
end

function destroy_etc(backend::GPUBackend, state::DropoutLayerState)
  cuda_rand_states, input_copy = state.etc
  CUDA.free(cuda_rand_states)
  destroy(input_copy)
end

function forward(backend::GPUBackend, state::DropoutLayerState, inputs::Vector{Blob})
  # make copy of input blob
  copy!(state.etc[2], inputs[1])

  len = length(inputs[1])
  x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
  data_type = eltype(inputs[1])
  if data_type == Float32
    kernel = backend.mocha.dropout_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.dropout_forward_double
  end

  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
      (state.etc[1], length(inputs[1]), inputs[1].ptr.p,
      state.rand_vals.ptr.p, state.ratio, state.scale))
end

function backward(backend::GPUBackend, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    len = length(inputs[1])
    x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
    data_type = eltype(inputs[1])
    if data_type == Float32
      kernel = backend.mocha.dropout_backward_float
    elseif data_type == Float64
      kernel = backend.mocha.dropout_backward_double
    end

    CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
        (state.etc[1], length(inputs[1]), diffs[1].ptr.p,
        state.rand_vals.ptr.p, state.ratio, state.scale))

    # restore the input blob
    copy!(inputs[1], state.etc[2])
  end
end
