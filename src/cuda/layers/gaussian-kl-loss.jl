
function setup(backend::GPUBackend, layer::GaussianKLLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  aux_ones = make_blob(backend, data_type, size(inputs[2])...) # for summing
  fill!(aux_ones, 1.0)
  tmp = make_blob(backend, data_type, size(inputs[2])...)
  state = GaussianKLLossLayerState(layer, zero(data_type), zero(data_type), 0,
                                   @compat(Dict(:aux_ones => aux_ones, :tmp => tmp)))
  return state
end


function forward(backend::GPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob})
  mu  = inputs[1]
  sigma = inputs[2]
  num = get_num(mu)
  data_type = eltype(mu)
  n = length(mu) # length or num?

  aux_ones  = state.tmp_blobs[:aux_ones]
  log_sigma_tmp = state.tmp_blobs[:tmp]
  copy!(log_sigma_tmp, sigma)
  CuVec.log!(backend, log_sigma_tmp)

  Σμ²  = CuBLAS.dot(backend.cublas_ctx, data_type, n, mu.ptr, 1, mu.ptr, 1)
  Σσ²  = CuBLAS.dot(backend.cublas_ctx, data_type, n, sigma.ptr, 1, sigma.ptr, 1)
  logΣ = CuBLAS.dot(backend.cublas_ctx, data_type, n, log_sigma_tmp.ptr, 1, aux_ones.ptr, 1)
  state.loss = 0.5(Σμ² + Σσ² - 2logΣ - n) * state.layer.weight / num

  # accumulate statistics
  state.loss_accum *= state.n_accum
  state.loss_accum += state.loss * n
  state.loss_accum /= state.n_accum + n

  state.n_accum += n
end


function backward(backend::GPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})

  mu  = inputs[1]
  sigma = inputs[2]
  data_type = eltype(mu)
  num = get_num(mu)

  # diff = df/dmu[i] = mu
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    copy!(diff, mu)
    CuVec.mul_scal!(backend, diff, convert(data_type, state.layer.weight/num))
  end

  # diff = df/dsigma[i] = sigma - 1/sigma
  diff = diffs[2]

  tmp = state.tmp_blobs[:tmp]
  copy!(tmp, sigma)
  if isa(diff, CuTensorBlob)
    copy!(diff, sigma)
    copy!(tmp, sigma)
    CuVec.pow!(backend, tmp, convert(data_type, -1.0))
    CuVec.sub!(backend, diff, tmp)
    CuVec.mul_scal!(backend, diff, convert(data_type, state.layer.weight/num))
  end
end
