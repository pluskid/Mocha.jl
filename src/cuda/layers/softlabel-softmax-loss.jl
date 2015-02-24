function setup_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  state.etc = make_blob(backend, eltype(inputs[1]), size(inputs[1]))
end
function shutdown_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState)
  destroy(state.etc)
end
function forward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  forward(backend, state.softmax, Blob[pred])
  copy!(state.etc, state.softmax.blobs[1])
  prob = state.etc

  dims = size(prob)
  data_type = eltype(prob)

  CuVec.log!(backend, prob)
  loss = -CuBLAS.dot(backend.cublas_ctx, data_type, length(prob), prob.ptr, 1, label.ptr, 1)
  state.loss = loss / (prod(dims) / dims[state.op_dim])
end

function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    label = inputs[2]
    copy!(diff, state.softmax.blobs[1])
    data_type = eltype(diff)
    dims = size(diff)

    CuBLAS.axpy(backend.cublas_ctx, length(diff), -one(data_type), label.ptr, 1, diff.ptr, 1)
    CuBLAS.scal(backend.cublas_ctx, length(diff), convert(data_type, dims[state.op_dim]/prod(dims)), diff.ptr, 1)


    #spatial_dim, channels, num = split_dims(diff, state.softmax_loss.logistic.op_dim)
    #prob_dim = channels

    #x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X))
    #y_block = spatial_dim
    #z_block = prob_dim

    #if data_type == Float32
    #  kernel = backend.mocha.softlabel_softmax_loss_backward_float
    #elseif data_type == Float64
    #  kernel = backend.mocha.softlabel_softmax_loss_backward_double
    #else
    #  error("Unsupported data type $data_type")
    #end
    #CUDA.launch(kernel, (x_block, y_block, z_block), (CUDA.THREADS_PER_BLOCK_X, 1, 1),
    #    (diff.ptr.p, inputs[2].ptr.p, num, spatial_dim, prob_dim))
    #CuBLAS.scal(backend.cublas_ctx, length(diff), convert(data_type, 1.0/(spatial_dim*num)),
    #    diff.ptr, 1)
  end
end

