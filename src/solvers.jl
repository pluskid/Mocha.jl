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

function forward(net::Net)
  obj_val = 0.0

  for i = 1:length(net.layers)
    forward(net.sys, net.states[i], net.blobs_forward[i])

    if isa(net.layers[i], LossLayer)
      obj_val += net.states[i].blobs[1].data[1]
    end

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        obj_val += forward(net.sys, param.regularizer, param.blob)
      end
    end
  end

  return obj_val
end

function backward(net::Net)
  for i = length(net.layers):-1:1
    backward(net.sys, net.states[i], net.blobs_forward[i], net.blobs_backward[i])

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        backward(net.sys, param.regularizer, param.blob, param.gradient)
      end
    end
  end
end

function solver_task(net::Net{CPU})
  function solver_loop()
    init(net)

    for iter = 1:net.sys.max_iter
      obj_val = forward(net)
      backward(net)

      # switch to the actual solver co-routine
      produce(iter)

      if iter % 100 == 0
        @printf("%06d objective function = %f\n", iter, obj_val)
      end
    end

    # loop ended
    produce(0)
  end

  return Task(solver_loop)
end

include("solvers/sgd.jl")
