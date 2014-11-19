function forward(sys::System{CuDNNBackend}, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred      = inputs[1]
  label     = inputs[2]
  data_type = eltype(pred)

  width, height, channels, num = size(pred)

  spatial_dim = height*width
  prob_dim = channels

  x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X))
  y_block = spatial_dim

  loss_blob = make_zero_blob(sys.backend, Float32, 1, 1, 1, 1)

  if data_type == Float32
    kernel = sys.backend.mocha.logistic_loss_forward_float
  elseif data_type == Float64
    kernel = sys.backend.mocha.logistic_loss_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
      (pred.ptr.p, label.ptr.p, num, spatial_dim, prob_dim, loss_blob.ptr.p))

  loss = Float32[0]
  copy!(loss, loss_blob)
  state.loss = loss[1] / (spatial_dim * num)
end

