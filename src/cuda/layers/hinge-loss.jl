function forward(backend::GPUBackend, state::HingeLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  x_block = div(n + CUDA.THREADS_PER_BLOCK_X-1, CUDA.THREADS_PER_BLOCK_X)

  if length(state.loss_blob) < x_block
    destroy(state.loss_blob)
    state.loss_blob = make_blob(backend, data_type, x_block)
  end

  if data_type == Float32
    kernel = backend.mocha.hinge_loss_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.hinge_loss_forward_double
  else
    error("Unsupported data type $data_type")
  end

  CUDA.launch(kernel, (x_block,1), (CUDA.THREADS_PER_BLOCK_X, 1),
              (pred.ptr.p, label.ptr.p, n, state.loss_blob.ptr.p))

  losses = Array{data_type}(size(state.loss_blob)...)
  copy!(losses, state.loss_blob)
  state.loss = state.layer.weight * sum(losses[1:x_block]) / get_num(pred)

  # accumulate statistics
  state.loss_accum = (state.loss_accum*state.n_accum + state.loss*get_num(pred)) / (state.n_accum+get_num(pred))
  state.n_accum += get_num(pred)
end

function backward(backend::GPUBackend, state::HingeLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  const pred = inputs[1]
  const label = inputs[2]

  const data_type = eltype(pred)
  const n = length(pred)
  const neg_weight :: data_type = -state.layer.weight / get_num(pred)

  const x_block = div(n + CUDA.THREADS_PER_BLOCK_X-1, CUDA.THREADS_PER_BLOCK_X)

  if data_type == Float32
    kernel = backend.mocha.hinge_loss_backward_float
  elseif data_type == Float64
    kernel = backend.mocha.hinge_loss_backward_double
  else
    error("Unsupported data type $data_type")
  end

  const gradient1 :: CuPtr = isa(diffs[1], CuTensorBlob) ? diffs[1].ptr : CuPtr()
  const gradient2 :: CuPtr = isa(diffs[2], CuTensorBlob) ? diffs[2].ptr : CuPtr()

  CUDA.launch(kernel, (x_block,1), (CUDA.THREADS_PER_BLOCK_X,1),
              (pred.ptr.p, label.ptr.p, gradient1.p, gradient2.p, n, neg_weight))
end
