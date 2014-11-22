function setup_etc(sys::System{CuDNNBackend}, layer::AccuracyLayer, inputs)
  width, height, channels, num = size(inputs[1])
  etc = make_blob(sys.backend, eltype(inputs[1]), (width,height,1,num))
  return etc
end
function shutdown(sys::System{CuDNNBackend}, state::AccuracyLayerState)
  destroy(state.etc)
end

function forward(sys::System{CuDNNBackend}, state::AccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  width, height, channels, num = size(pred)
  spatial_dim = width*height
  data_type = eltype(pred)

  x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X));
  y_block = int(ceil(float64(spatial_dim)/CUDA.THREADS_PER_BLOCK_Y));

  if data_type == Float32
    kernel = sys.backend.mocha.accuracy_forward_float
  elseif data_type == Float64
    kernel = sys.backend.mocha.accuracy_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, (x_block,y_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y),
      (pred.ptr.p, label.ptr.p, state.etc.ptr.p, num, channels, spatial_dim));

  N = num * spatial_dim
  accuracy = CuBLAS.dot(sys.backend.cublas_ctx, data_type, N, state.etc.ptr, 1, state.etc.ptr, 1)

  # accumulate accuracy
  state.accuracy = (state.accuracy * state.n_accum + accuracy) / (N + state.n_accum)
  state.n_accum += N
end

