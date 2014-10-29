type SGD <: Solver
end

function solve(sgd::SGD, net::Net{CPU})
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
