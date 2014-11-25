function backward(sys::System{CuDNNBackend}, state::SoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    copy!(diff, state.softmax.blobs[1])

    data_type = eltype(diff)
    height, width, channels, num = size(diff)

    spatial_dim = height*width
    prob_dim = channels

    x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X))
    y_block = spatial_dim

    if isa(state.logistic.weights_blob, NullBlob)
      weights = convert(Ptr{data_type}, 0)
    else
      weights = state.logistic.weights_blob.ptr.p
    end

    if data_type == Float32
      kernel = sys.backend.mocha.softmax_loss_backward_float
    elseif data_type == Float64
      kernel = sys.backend.mocha.softmax_loss_backward_double
    else
      error("Unsupported data type $data_type")
    end
    CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK_X, 1),
        (diff.ptr.p, inputs[2].ptr.p, weights, num, spatial_dim, prob_dim))
    CuBLAS.scal(sys.backend.cublas_ctx, length(diff), convert(data_type, 1.0/(spatial_dim*num)),
        diff.ptr, 1)
  end
end

