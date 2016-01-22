
function setup_etc(backend::GPUBackend, layer::RandomNormalLayer)
  cuda_rand_states = CuPtr
  kernel = backend.mocha.stdnormal_init
  rnd_state_size_blob = make_blob(backend, Float64, 1, 1, 1, 1)
  CUDA.launch(backend.mocha.stdnormal_alloc_size, 1, 1, (rnd_state_size_blob.ptr.p, ))
  rnd_state_size = Float64[0]
  copy!(rnd_state_size, rnd_state_size_blob)
  destroy(rnd_state_size_blob)
  rnd_state_size = round(Int, rnd_state_size[1])

  etc = Any[]
  outlen = prod(layer.output_dims)
  for i = 1:length(layer.tops)
      len = outlen*layer.batch_sizes[i]
      cuda_rand_states = CUDA.cualloc(UInt8, rnd_state_size*len)
      x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
      seed = rand(UInt)
      println("launching stdnormal init on bloc $i with len $len")
      CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (cuda_rand_states, seed))
      push!(etc, cuda_rand_states)
  end
  return etc
end

function destroy_etc(backend::GPUBackend, state::RandomNormalLayerState)
    for i = 1:length(state.etc)
        CUDA.free(state.etc[i])
    end
end

function forward(backend::GPUBackend, state::RandomNormalLayerState, inputs::Vector{Blob})
    for i = 1:length(state.blobs)
        len = length(state.blobs[i])
        x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
        data_type = state.layer.eltype
        if data_type == Float32
            kernel = backend.mocha.stdnormal_forward_float
        elseif data_type == Float64
            kernel = backend.mocha.stdnormal_forward_double
        end

        CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X,
                    (state.etc[i], state.blobs[i].ptr.p, len,))
    end
end

function backward(backend::GPUBackend, state::RandomNormalLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})

end

