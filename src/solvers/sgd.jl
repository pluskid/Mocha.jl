type SGD <: Solver
  maxiter :: Int
  learning_rate
end

function solve(sgd::SGD, net::Net)
  # initialize
  # TODO: properly initialize network parameters

  # loop
  #   forward
  #   backward
  #   update
  for iter = 1:sgd.maxiter
    for i = 1:length(net.layers)
      forward(net.states[i], net.blobs_forward[i])
    end

    for i = length(net.layers):-1:1
      backward(net.states[i], net.blobs_forward[i], net.blobs_backward[i])
    end

    for i = 1:length(net.layers)
      state = net.states[i]
      if :parameters ∈ names(state)
        for j = 1:length(state.parameters)
          state.parameters[j].data -= sgd.learning_rate * state.gradients[j].data
        end
      end
    end

    if iter % 1 == 0
      for i = 1:length(net.layers)
        if isa(net.layers[i], LossLayer)
          @printf("%06d %s = %f\n", iter, net.layers[i].name, net.states[i].blobs[1].data[1])
        end
      end
    end
  end
end
