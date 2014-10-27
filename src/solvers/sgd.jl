type SGD <: Solver
  maxiter :: Int
  learning_rate
  momentum
end

function solve(sgd::SGD, net::Net{CPU})
  init(net)

  param_history = Array(Vector{Array}, length(net.layers))
  for i = 1:length(net.states)
    state = net.states[i]
    if :parameters ∈ names(state)
      param_history[i] = [zeros(eltype(x.blob.data),size(x.blob.data)) for x in state.parameters]
    end
  end

  for iter = 1:sgd.maxiter
    for i = 1:length(net.layers)
      forward(net.sys, net.states[i], net.blobs_forward[i])
    end

    for i = length(net.layers):-1:1
      backward(net.sys, net.states[i], net.blobs_forward[i], net.blobs_backward[i])
    end

    for i = 1:length(net.layers)
      state = net.states[i]
      if :parameters ∈ names(state)
        for j = 1:length(state.parameters)
          param_history[i][j] *= sgd.momentum
          param_history[i][j] -= sgd.learning_rate * state.parameters[j].gradient.data
          state.parameters[j].blob.data += param_history[i][j]
        end
      end
    end

    if iter % 100 == 0
      for i = 1:length(net.layers)
        if isa(net.layers[i], LossLayer)
          @printf("%06d %s = %f\n", iter, net.layers[i].name, net.states[i].blobs[1].data[1])
        end
      end
    end
  end
end
