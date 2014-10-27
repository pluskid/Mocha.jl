export Solver
export SGD

export solve

abstract Solver

############################################################
# General utilities that could be used by all solvers
############################################################
# Initialize network parameters according to defined initializers
function init(net::Net)
  for i = 1:length(net.layers)
    state = net.states[i]
    if :parameters ∈ names(state)
      for j = 1:length(state.parameters)
        init(state.parameters[j].initializer, state.parameters[j].blob)
      end
    end
  end
end

function solver_task(net::Net{CPU})
  function solver_loop()
    init(net)

    for iter = 1:net.sys.max_iter
      for i = 1:length(net.layers)
        forward(net.sys, net.states[i], net.blobs_forward[i])
      end

      for i = length(net.layers):-1:1
        backward(net.sys, net.states[i], net.blobs_forward[i], net.blobs_backward[i])
      end

      # switch to the actual solver co-routine
      produce(iter)

      if iter % 100 == 0
        for i = 1:length(net.layers)
          if isa(net.layers[i], LossLayer)
            @printf("%06d %s = %f\n", iter, net.layers[i].name, net.states[i].blobs[1].data[1])
          end
        end
      end
    end

    # loop ended
    produce(0)
  end

  return Task(solver_loop)
end

include("solvers/sgd.jl")
