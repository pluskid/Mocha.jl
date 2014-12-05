function setup_etc(backend::GPUBackend, layer::DropoutLayer, inputs::Vector{Blob})
  cuda_rand_states = CuPtr
  kernel = backend.mocha.dropout_init
  rnd_state_size_blob = make_blob(backend, Float64, 1, 1, 1, 1)
  CUDA.launch(backend.mocha.dropout_alloc_size, 1, 1, (rnd_state_size_blob.ptr.p, ))
  rnd_state_size = Float64[0]
  copy!(rnd_state_size, rnd_state_size_blob)
  destroy(rnd_state_size_blob)
  rnd_state_size = int(rnd_state_size[1])

  len = length(inputs[1])
  cuda_rand_states = CUDA.cualloc(Uint8, rnd_state_size*len)
  x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (cuda_rand_states, len))

  return cuda_rand_states
end

function destroy_etc(backend::GPUBackend, state::DropoutLayerState)
  CUDA.free(state.etc)
end

function forward(backend::GPUBackend, state::DropoutLayerState, inputs::Vector{Blob})
  len = length(inputs[1])
  x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
  data_type = eltype(inputs[1])
  if data_type == Float32
    kernel = backend.mocha.dropout_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.dropout_forward_double
  end

  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
      (state.etc, length(inputs[1]), inputs[1].ptr.p,
      state.rand_vals.ptr.p, state.ratio, state.scale))
end

function backward(backend::GPUBackend, state::DropoutLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    len = length(inputs[1])
    x_block = int(ceil(float64(len)/CUDA.THREADS_PER_BLOCK_X))
    data_type = eltype(inputs[1])
    if data_type == Float32
      kernel = backend.mocha.dropout_backward_float
    elseif data_type == Float64
      kernel = backend.mocha.dropout_backward_double
    end

    CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
        (state.etc, length(inputs[1]), diffs[1].ptr.p,
        state.rand_vals.ptr.p, state.ratio, state.scale))
  end
end
