
const log2π = log(2π)

function forward(backend::GPUBackend, state::GaussianKLLossLayerState, inputs::Vector{Blob})
  mu  = inputs[1]
  sigma = inputs[2]

  data_type = eltype(mu)
  n = length(mu) # length or num?
  #@assert length(sigma) == n
  state.loss = -0.5(n * log2π  + sum(mu.data.^2 + sigma.data.^2)) * state.layer.weight

end
