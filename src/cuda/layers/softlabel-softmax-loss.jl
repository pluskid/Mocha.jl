function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    pred = inputs[1]
    label = inputs[2]
    data_type = eltype(pred)
    copy!(state.softmax_loss.logistic.weights_blob, label)
    erase!(diff)

    for i = 1:length(state.fake_labels)
      backward(backend, state.softmax_loss, Blob[pred, state.fake_labels[i]], Blob[state.fake_diff])
      CuBLAS.axpy(backend.cublas_ctx, length(pred), one(data_type), state.fake_diff.ptr, 1, diff.ptr, 1)
    end
  end
end

