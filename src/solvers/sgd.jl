immutable SGD <: Solver
  params        :: SolverParameters
  coffee_lounge :: CoffeeLounge

  SGD(params::SolverParameters) = new(params, CoffeeLounge())
end

type SGDInternalState <: SolverInternelState
  param_states  :: Vector{LayerState}
  param_history :: Vector{Vector{Blob}}
  last_momentum :: Float64
end

function setup(sgd::SGD, net::Net, solver_state::SolverState)
  param_states  = map(i -> net.states[i],
      filter(i -> isa(net.layers[i], TrainableLayer), 1:length(net.layers)))
  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end

  return SGDInternalState(param_states, param_history, solver_state.momentum)
end

function update(sgd::SGD, net::Net, i_state::SGDInternalState, solver_state::SolverState)
  for i = 1:length(i_state.param_states)
    state   = i_state.param_states[i]
    history = i_state.param_history[i]
    for j = 1:length(state.parameters)
      hist_blob = history[j]
      gradient  = state.parameters[j].gradient
      data_type = eltype(hist_blob)

      # to keep iteration <-> momentum correspondence consistent with the Nesterov solver,
      # we use the momentum from the last iteration
      update_parameters(net, sgd, state.parameters[j].learning_rate * solver_state.learning_rate,
          i_state.last_momentum, state.parameters[j].blob, hist_blob, gradient, data_type)
    end
  end

  i_state.last_momentum = solver_state.momentum
end

function shutdown(sgd::SGD, i_state::SGDInternalState)
  map(x -> map(destroy, x), i_state.param_history)
end

function update_parameters(net::Net{CPUBackend}, solver::SGD, learning_rate, momentum, param_blob, hist_blob, gradient, data_type)
  # hist_blob = momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, momentum), pointer(hist_blob.data), 1)
  # hist_blob = -learning_rate * gradient + hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, -learning_rate), pointer(gradient.data), 1, pointer(hist_blob.data), 1)

  # update parameter
  # param_blob += hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, 1), pointer(hist_blob.data), 1, pointer(param_blob.data), 1)
end

