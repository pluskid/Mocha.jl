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
    learning_rate = get_learning_rate(sgd.params.lr_policy, sgd.params.base_lr, solver_state)

    # update parameters
    for i = 1:length(param_states)
      state = param_states[i]
      history = param_history[i]
      for j = 1:length(state.parameters)
        blob = history[j]
        gradient = state.parameters[j].gradient
        data_type = eltype(blob)

        update_parameters(net, sgd, learning_rate, state, state.parameters[j].blob,
            blob, gradient, data_type)
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

function update_parameters(net::Net{CPUBackend}, solver, learning_rate, state, param_blob, blob, gradient, data_type)
  # blob = momentum * blob
  BLAS.scal!(length(blob), convert(data_type, solver.params.momentum), blob.data, 1)
  # blob = - learning_rate * gradient + blob
  BLAS.axpy!(length(blob), convert(data_type, -learning_rate), gradient.data, 1, blob.data, 1)

  # update parameter
  # param_blob += blob
  BLAS.axpy!(length(blob), convert(data_type, 1), blob.data, 1, param_blob.data, 1)
end
function update_parameters(net::Net{CuDNNBackend}, solver, learning_rate, state, param_blob, blob, gradient, data_type)
  # blob = net.sys.momentum * blob
  CuBLAS.scal(net.sys.backend.cublas_ctx, length(blob), convert(data_type, solver.params.momentum),
      blob.ptr, 1)
  # blob = - net.sys.learning_rate * gradient + blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, -learning_rate),
      gradient.ptr, 1, blob.ptr, 1)

  # update parameter
  # param_blob += blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, 1),
      blob.ptr, 1, param_blob.ptr, 1)
end
