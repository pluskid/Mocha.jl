function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    copy!(diff, state.softmax_loss.softmax.blobs[1])

    data_type = eltype(diff)
    spatial_dim, channels, num = split_dims(diff, state.softmax_loss.logistic.op_dim)
    prob_dim = channels

    x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X))
    y_block = spatial_dim
    z_block = prob_dim

    if data_type == Float32
      kernel = backend.mocha.softlabel_softmax_loss_backward_float
    elseif data_type == Float64
      kernel = backend.mocha.softlabel_softmax_loss_backward_double
    else
      error("Unsupported data type $data_type")
    end
    CUDA.launch(kernel, (x_block, y_block, z_block), (CUDA.THREADS_PER_BLOCK_X, 1, 1),
        (diff.ptr.p, inputs[2].ptr.p, num, spatial_dim, prob_dim))
    CuBLAS.scal(backend.cublas_ctx, length(diff), convert(data_type, 1.0/(spatial_dim*num)),
        diff.ptr, 1)
  end
end

