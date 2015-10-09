function setup_etc(backend::GPUBackend, layer::AccuracyLayer, op_dim::Int, inputs)
  dims = [size(inputs[1])...]
  dims[op_dim] = 1
  etc = make_blob(backend, eltype(inputs[1]), dims...)
  return etc
end
function shutdown(backend::GPUBackend, state::AccuracyLayerState)
  destroy(state.etc)
end

function forward(backend::GPUBackend, state::AccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  spatial_dim, pred_dim, num = split_dims(pred, state.op_dim)
  data_type = eltype(pred)

  x_block = round(Int, ceil(convert(Float64, num)/CUDA.THREADS_PER_BLOCK_X));
  y_block = round(Int, ceil(convert(Float64, spatial_dim)/CUDA.THREADS_PER_BLOCK_Y));

  if data_type == Float32
    kernel = backend.mocha.accuracy_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.accuracy_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, (x_block,y_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y),
      (pred.ptr.p, label.ptr.p, state.etc.ptr.p, num, pred_dim, spatial_dim));

  N = num * spatial_dim
  accuracy = CuBLAS.dot(backend.cublas_ctx, data_type, N, state.etc.ptr, 1, state.etc.ptr, 1)

  # accumulate accuracy
  state.accuracy = (state.accuracy * state.n_accum + accuracy) / (N + state.n_accum)
  state.n_accum += N
end

