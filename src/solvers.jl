export SolverParameters
export SGD

export LearningRatePolicy, LRPolicy, get_learning_rate

export add_coffee_break, solve

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
    base_lr * (1 + policy.gamma * state.iter) ^ (-policy.power)


@defstruct SolverParameters Any (
  (base_lr :: FloatingPoint = 0.01, base_lr > 0),
  lr_policy :: LearningRatePolicy = LRPolicy.Fixed(),
  (momentum :: FloatingPoint = 0.9, 0 <= momentum < 1),
  (max_iter :: Int = 0, max_iter > 0),
  (regu_coef :: FloatingPoint = 0.0005, regu_coef >= 0),
)

############################################################
# Coffee break utilities
############################################################
function add_coffee_break(solver::Solver, coffee::Coffee; every_n_iter::Int=0, every_n_epoch::Int=0)
  cb = CoffeeBreak(coffee, every_n_iter, every_n_epoch)
  push!(solver.coffee_breaks, cb)
end
function init_coffee_breaks(solver::Solver, net::Net)
  for cb in solver.coffee_breaks
    init(cb.coffee, net)
  end
end
function check_coffee_breaks(solver::Solver, state::SolverState, net::Net)
  for cb in solver.coffee_breaks
    check_coffee_break(cb, state, net)
  end
end
function destroy_coffee_breaks(solver::Solver, net::Net)
  for cb in solver.coffee_breaks
    destroy(cb.coffee, net)
  end
end

############################################################
# General utilities that could be used by all solvers
############################################################
function update_solver_state(state::SolverState, obj_val :: Float64)
  state.iter += 1
  state.obj_val = obj_val
end
function stop_condition_satisfied(solver::Solver, state::SolverState, net::Net)
  if state.iter > solver.params.max_iter
    return true
  end
  return false
end

include("solvers/sgd.jl")
