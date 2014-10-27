type SGD <: Solver
end

function solve(sgd::SGD, net::Net{CPU})
  loop = solver_task(net)

  param_history = Array(Vector{Array}, length(net.layers))
  for i = 1:length(net.states)
    state = net.states[i]
    if :parameters ∈ names(state)
      param_history[i] = [zeros(eltype(x.blob.data),size(x.blob.data)) for x in state.parameters]
    end
  end

  while consume(loop) > 0
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
  end
end
