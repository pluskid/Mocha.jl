type SGD <: Solver
  params        :: SolverParameters
  coffee_breaks :: Vector{CoffeeBreak}

  SGD(params::SolverParameters) = new(params, CoffeeBreak[])
end

function solve(sgd::SGD, net::Net)
  param_states = filter(x -> :parameters ∈ names(x), net.states)

  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.sys.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end

  init(net)
  init_coffee_breaks(sgd, net)
  solver_state = SolverState(0, 0.0)

  while true
    check_coffee_breaks(CoffeeBreakTime.Morning(), sgd, solver_state, net)

    obj_val = forward_backward(net)
    learning_rate = get_learning_rate(sgd.params.lr_policy, solver_state)

    # update parameters
    for i = 1:length(param_states)
      state = param_states[i]
      history = param_history[i]
      for j = 1:length(state.parameters)
        hist_blob = history[j]
        gradient = state.parameters[j].gradient
        data_type = eltype(hist_blob)

        update_parameters(net, sgd, state.parameters[j].learning_rate * learning_rate, 
            state, state.parameters[j].blob, hist_blob, gradient, data_type)
      end
    end

    update_solver_state(solver_state, obj_val)
    check_coffee_breaks(CoffeeBreakTime.Evening(), sgd, solver_state, net)

    if stop_condition_satisfied(sgd, solver_state, net)
      break
    end
  end

  destroy_coffee_breaks(sgd, net)
end

function update_parameters(net::Net{CPUBackend}, solver::SGD, learning_rate, state, param_blob, hist_blob, gradient, data_type)
  # hist_blob = momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, solver.params.momentum), hist_blob.data, 1)
  # hist_blob = - learning_rate * gradient + hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, -learning_rate), gradient.data, 1, hist_blob.data, 1)

  # update parameter
  # param_blob += hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, 1), hist_blob.data, 1, param_blob.data, 1)
end
