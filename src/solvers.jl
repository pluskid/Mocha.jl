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
      for param in state.parameters
        init(param.initializer, param.blob)

        # scale per-layer regularization coefficient globally
        param.regularizer.coefficient *= net.sys.regularization_coef
      end
    end
  end
end
function prepare_iteration(iter::Int, net::Net)
  obj_val = forward(net)
  backward(net)

  if iter % 1 == 0
    @printf("%06d objective function = %f\n", iter, obj_val)
  end
end
function finalize_iteration(iter::Int, net::Net)
  if iter > net.sys.max_iter
    return false
  end
  return true
end


function forward(net::Net)
  obj_val = 0.0

  for i = 1:length(net.layers)
    forward(net.sys, net.states[i], net.blobs_forward[i])
    if :neuron ∈ names(net.layers[i]) && !isa(net.layers[i].neuron, Neurons.Identity)
      for blob in net.states[i].blobs
        forward(net.sys, net.layers[i].neuron, blob)
      end
    end

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
    if :neuron ∈ names(net.layers[i]) && !isa(net.layers[i].neuron, Neurons.Identity)
      state = net.states[i]
      for j = 1:length(state.blobs)
        backward(net.sys, net.layers[i].neuron, state.blobs[j], state.blobs_diff[j])
      end
    end
    backward(net.sys, net.states[i], net.blobs_forward[i], net.blobs_backward[i])

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        backward(net.sys, param.regularizer, param.blob, param.gradient)
      end
    end
  end
end

include("solvers/sgd.jl")
