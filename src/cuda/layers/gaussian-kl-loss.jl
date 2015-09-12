
const log2π = log(2π)

function forward(backend::GPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob})
  mu  = inputs[1]
  sigma = inputs[2]
  num = get_num(mu)
  data_type = eltype(mu)
  n = length(mu) # length or num?


  Σμ² = CuBLAS.dot(backend.cublas_ctx, data_type, n, mu.ptr, 1, mu.ptr, 1)
  Σσ² = CuBLAS.dot(backend.cublas_ctx, data_type, n, sigma.ptr, 1, sigma.ptr, 1)

  state.loss = -0.5(n * log2π + Σμ² + Σσ²) * -state.layer.weight / num
end


function backward(backend::GPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})

  mu  = inputs[1]
  sigma = inputs[2]
  data_type = eltype(mu)
  num = get_num(mu)
    #diff = df/dmu[i]
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    copy!(diff, mu)
    CuVec.mul_scal!(backend, diff, convert(data_type, state.layer.weight/num))
  end

    #diff = df/dsigma[i]
  diff = diffs[2]
  if isa(diff, CuTensorBlob)
    copy!(diff, sigma)
    CuVec.mul_scal!(backend, diff, convert(data_type, state.layer.weight/num))
  end
end
