# An implementation of Nesterov's Accelerated method based on a reformulation
# that makes it similar to ordinary (Stochastic) Gradient Descent with Momentum
#
# Yoshua Bengio, Nicolas Boulanger-Lewandowski, Razvan Pascanu. Advances in
# Optimizing Recurrent Networks. arXiv:1212.0901 [cs.LG]

immutable Nesterov <: Solver
  params        :: SolverParameters
  coffee_lounge :: CoffeeLounge

  Nesterov(params::SolverParameters) = new(params, CoffeeLounge())
end

type NesterovInternalState <: SolverInternelState
  param_states  :: Vector{LayerState}
  param_history :: Vector{Vector{Blob}}
  last_momentum :: Float64
end

function setup(nag::Nesterov, net::Net, solver_state::SolverState)
  param_states  = map(i -> net.states[i],
      filter(i -> has_param(net.layers[i]) && !is_frozen(net.states[i]), 1:length(net.layers)))
  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end

  return NesterovInternalState(param_states, param_history, solver_state.momentum)
end

function update(nag::Nesterov, net::Net, i_state::NesterovInternalState, solver_state::SolverState)
  for i = 1:length(i_state.param_states)
    state   = i_state.param_states[i]
    history = i_state.param_history[i]
    for j = 1:length(state.parameters)
      hist_blob = history[j]
      gradient  = state.parameters[j].gradient
      data_type = eltype(hist_blob)

      update_parameters(net, nag, state.parameters[j].learning_rate * solver_state.learning_rate,
          i_state.last_momentum, solver_state.momentum, state.parameters[j].blob, hist_blob, gradient, data_type)
    end
  end

  i_state.last_momentum = solver_state.momentum
end


function shutdown(nag::Nesterov, i_state::NesterovInternalState)
  map(x -> map(destroy, x), i_state.param_history)
end

function update_parameters(net::Net{CPUBackend}, solver::Nesterov, learning_rate,
    last_momentum, momentum, param_blob, hist_blob, gradient, data_type)

  # param_blob += -last_momentum* hist_blob (update with vt-1)
  BLAS.axpy!(length(hist_blob), convert(data_type, -last_momentum), hist_blob.data, 1, param_blob.data, 1)

  # hist_blob = last_momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, last_momentum), hist_blob.data, 1)

  # hist_blob = -learning_rate * gradient + hist_blob (calc vt)
  BLAS.axpy!(length(hist_blob), convert(data_type, -learning_rate), gradient.data, 1, hist_blob.data, 1)


  # param_blob += (1+momentum) * hist_blob (update with vt)
  BLAS.axpy!(length(hist_blob), convert(data_type, 1 + momentum), hist_blob.data, 1, param_blob.data, 1)
end

