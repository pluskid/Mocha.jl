#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
type CuBinaryAccuracyEtcState{T}
  n_wrong  :: SyncMem{T}
  N      :: Int
end
function setup_etc(backend::GPUBackend, layer::BinaryAccuracyLayer, inputs)
  n_wrong_blob = make_blob(backend, Float32, (1,))
  n_wrong = SyncMem(backend, n_wrong_blob)
  return CuBinaryAccuracyEtcState(n_wrong, 0)
end
function shutdown(backend::GPUBackend, state::BinaryAccuracyLayerState)
  custate = state.etc
  destroy(custate.n_wrong)
end

function forward(backend::GPUBackend, state::BinaryAccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]
  custate = state.etc

  const N = length(pred)
  const x_block = div(N+CUDA.THREADS_PER_BLOCK_X-1, CUDA.THREADS_PER_BLOCK_X)

  const data_type = eltype(pred)
  if data_type == Float32
    kernel = get_mocha(backend).binary_accuracy_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).binary_accuracy_forward_double
  else
    error("Unsupported data type $data_type")
  end

  threshold = convert(data_type, state.layer.threshold)
  erase!(custate.n_wrong.dev_blob)
  CUDA.launch(kernel, (x_block,1),(CUDA.THREADS_PER_BLOCK_X,1),
      (get_ptr(pred).p, get_ptr(label).p, N, threshold, get_ptr(custate.n_wrong.dev_blob).p),
      get_stream(backend));

  custate.N = N
end

function sync(backend::GPUBackend, state::BinaryAccuracyLayerState)
  custate = state.etc
  sync_all!(custate.n_wrong)

  # accumulate accuracy
  @assert length(custate.n_wrong.host_blob.data) == backend.dev_count
  state.n_wrong += sum(custate.n_wrong.host_blob.data)[1]
  state.n_accum += custate.N * backend.dev_count
  state.accuracy = (state.n_accum-state.n_wrong) / state.n_accum
end