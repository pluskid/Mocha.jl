type SGD <: Solver
end

function solve(sgd::SGD, net::Net{CPUBackend})
  param_history = Array(Vector{Array}, length(net.layers))
  for i = 1:length(net.states)
    state = net.states[i]
    if :parameters ∈ names(state)
      param_history[i] = [zeros(eltype(x.blob.data),size(x.blob.data)) for x in state.parameters]
    end
  end

  iter = 1
  init(net)
  while true
    prepare_iteration(iter, net)
    for i = 1:length(net.layers)
      state = net.states[i]
      if :parameters ∈ names(state)
        for j = 1:length(state.parameters)
          param_history[i][j] *= net.sys.momentum
          param_history[i][j] -= net.sys.learning_rate * state.parameters[j].gradient.data
          state.parameters[j].blob.data += param_history[i][j]
        end
      end
    end

    if !finalize_iteration(iter, net)
      break
    end
    iter += 1
  end
end

function solve(sgd::SGD, net::Net{CuDNNBackend})
  param_states = filter(x -> :params ∈ names(x), net.states)

  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [cudnn_make_pod_blob(eltype(x.blob),size(x.blob)) for x in state.parameters]
  end

  solver_state = init(net)
  while true
    forward_backward(solver_state, net)

    # update parameters
    for i = 1:length(param_states)
      state = param_states[i]
      history = param_history[i]
      for j = 1:length(state.parameters)
        blob = history[j]
        gradient = state.parameters[j].gradient
        data_type = eltype(blob)

        # blob = net.sys.moment * blob
        CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, net.sys.moment-1),
            blob.ptr, 1, blob.ptr, 1)
        # blob = - net.sys.learning_rate * gradient + blob
        CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, -net.sys.learning_rate),
            gradient.ptr, 1, blob.ptr, 1)

        # update parameter
        # state.parameters[j] += blob
        CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, 1),
            blob.ptr, 1, state.parameters[j].ptr, 1)
      end
    end

    if stop_condition_satisfied(solver_state, net)
      break
    end
  end
end
