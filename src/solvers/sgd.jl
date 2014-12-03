type SGD <: Solver
  params        :: SolverParameters
  coffee_lounge :: CoffeeLounge

  SGD(params::SolverParameters) = new(params, CoffeeLounge())
end

type SGDInternalState <: SolverInternelState
  param_states  :: Vector{LayerState}
  param_history :: Vector{Vector{Blob}}
end

function setup(sgd::SGD, net::Net)
  param_states  = filter(x -> :parameters ∈ names(x), net.states)
  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.sys.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end

  return SGDInternalState(param_states, param_history)
end

function update(sgd::SGD, net::Net, i_state::SolverInternelState, solver_state::SolverState)
  # update parameters
  for i = 1:length(i_state.param_states)
    state   = i_state.param_states[i]
    history = i_state.param_history[i]
    for j = 1:length(state.parameters)
      hist_blob = history[j]
      gradient  = state.parameters[j].gradient
      data_type = eltype(hist_blob)

      update_parameters(net, sgd, state.parameters[j].learning_rate * solver_state.learning_rate,
          solver_state.momentum, state.parameters[j].blob, hist_blob, gradient, data_type)

      # apply constraints after update
      cons_every = state.parameters[j].constraint.every_n_iter
      if cons_every > 0 && solver_state.iter % cons_every == 0
        constrain!(net.sys, state.parameters[j].constraint, state.parameters[j].blob)
      end
    end
  end
end

function shutdown(sgd::SGD, i_state::SGDInternalState)
  map(x -> map(destroy, x), i_state.param_history)
end

function update_parameters(net::Net{CPUBackend}, solver::SGD, learning_rate, momentum, param_blob, hist_blob, gradient, data_type)
  # hist_blob = momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, momentum), hist_blob.data, 1)
  # hist_blob = learning_rate * gradient + hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, learning_rate), gradient.data, 1, hist_blob.data, 1)

  # update parameter
  # param_blob += -hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, -1), hist_blob.data, 1, param_blob.data, 1)
end

