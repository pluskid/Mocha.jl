function forward(backend::GPUBackend, state::MultinomialLogisticLossLayerState, inputs::Vector{Blob})
  pred      = inputs[1]
  label     = inputs[2]
  data_type = eltype(pred)

  spatial_dim, channels, num = split_dims(pred, state.op_dim)
  prob_dim = channels

  x_block = round(Int64, ceil(convert(Float64, num)/CUDA.THREADS_PER_BLOCK_X))
  y_block = spatial_dim

  loss_blob = make_zero_blob(backend, Float32, 1, 1, 1, 1)

  if data_type == Float32
    kernel = backend.mocha.logistic_loss_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.logistic_loss_forward_double
  else
    error("Unsupported data type $data_type")
  end

  if isa(state.weights_blob, NullBlob)
    weights = convert(Ptr{data_type}, 0)
  else
    weights = state.weights_blob.ptr.p
  end

  CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
      (pred.ptr.p, label.ptr.p, weights, num, spatial_dim, prob_dim, loss_blob.ptr.p))

  loss = Float32[0]
  copy!(loss, loss_blob)
  state.loss = state.layer.weight * loss[1] / (spatial_dim * num)
  destroy(loss_blob)
end

