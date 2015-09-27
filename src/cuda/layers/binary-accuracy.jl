function setup_etc(backend::GPUBackend, layer::BinaryAccuracyLayer, inputs)
  etc = make_blob(backend, Float32, 1)
  return etc
end
function shutdown(backend::GPUBackend, state::BinaryAccuracyLayerState)
  destroy(state.etc)
end

function forward(backend::GPUBackend, state::BinaryAccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  const N = length(pred)
  const x_block = div(N+CUDA.THREADS_PER_BLOCK_X-1, CUDA.THREADS_PER_BLOCK_X)

  const data_type = eltype(pred)
  if data_type == Float32
    kernel = backend.mocha.binary_accuracy_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.binary_accuracy_forward_double
  else
    error("Unsupported data type $data_type")
  end

  threshold = convert(data_type, state.layer.threshold)
  erase!(state.etc)
  CUDA.launch(kernel, (x_block,1),(CUDA.THREADS_PER_BLOCK_X,1),
      (pred.ptr.p, label.ptr.p, N, threshold, state.etc.ptr.p));

  n_wrong = Float32[0.0f0]
  copy!(n_wrong, state.etc)
  state.n_wrong += n_wrong[1]

  # accumulate accuracy
  state.n_accum += N
  state.accuracy = (state.n_accum-state.n_wrong) / state.n_accum
end

