export SolverParameters, Solver, SolverState
export SGD

export LearningRatePolicy, LRPolicy, get_learning_rate

export solve

abstract Solver

type SolverState
  iter :: Int
end


############################################################
# Learning rate policy
############################################################
abstract LearningRatePolicy
module LRPolicy
using ..Mocha.LearningRatePolicy
type Fixed <: LearningRatePolicy end

# base_lr * gamma ^ (floor(iter / stepsize))
type Step <: LearningRatePolicy
  gamma :: FloatingPoint
  stepsize :: Int
end

# base_lr * gamma ^ iter
type Exp <: LearningRatePolicy 
  gamma :: FloatingPoint
end
type Inv <: LearningRatePolicy 
  gamma :: FloatingPoint
  power :: FloatingPoint
end
end # module LRPolicy
get_learning_rate(policy::LRPolicy.Fixed, base_lr, state::SolverState) = base_lr
get_learning_rate(policy::LRPolicy.Step, base_lr, state::SolverState) = 
    base_lr * policy.gamma ^ (floor(state.iter / policy.stepsize))
get_learning_rate(policy::LRPolicy.Exp, base_lr, state::SolverState) =
    base_lr * policy.gamma ^ state.iter
get_learning_rate(policy::LRPolicy.Inv, base_lr, state::SolverState) =
    base_lr * (1 + policy.gamma * state.iter) ^ (-state.power)


@defstruct SolverParameters Any (
  (base_lr :: FloatingPoint = 0.01, base_lr > 0),
  lr_policy :: LearningRatePolicy = LRPolicy.Fixed(),
  (momentum :: FloatingPoint = 0.9, 0 <= momentum < 1),
  (max_iter :: Int = 0, max_iter > 0),
  (regu_coef :: FloatingPoint = 0.0005, regu_coef >= 0),
)

############################################################
# General utilities that could be used by all solvers
############################################################
# Initialize network parameters according to defined initializers
function init(solver::Solver, net::Net)
  for i = 1:length(net.layers)
    state = net.states[i]
    if :parameters ∈ names(state)
      for param in state.parameters
        init(param.initializer, param.blob)

        # scale per-layer regularization coefficient globally
        param.regularizer.coefficient *= solver.params.regu_coef
      end
    end
  end

  return SolverState(0)
end
function forward_backward(state::SolverState, net::Net)
  obj_val = forward(net)
  backward(net)

  if state.iter % 100 == 0
    @printf("%06d objective function = %f\n", state.iter, obj_val)
  end
  state.iter += 1
end
function stop_condition_satisfied(solver::Solver, state::SolverState, net::Net)
  if state.iter > solver.params.max_iter
    return true
  end
  return false
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
      obj_val += net.states[i].loss
      #println("~~ obj_val = $obj_val")
    end

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        #println("== obj_val = $obj_val")
        obj_val += forward(net.sys, param.regularizer, param.blob)
        #println("!! obj_val = $obj_val")
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
