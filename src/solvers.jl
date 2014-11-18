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
type Fixed <: LearningRatePolicy 
  base_lr :: FloatingPoint
end

# base_lr * gamma ^ (floor(iter / stepsize))
type Step <: LearningRatePolicy
  base_lr  :: FloatingPoint
  gamma    :: FloatingPoint
  stepsize :: Int
end

# base_lr * gamma ^ iter
type Exp <: LearningRatePolicy 
  base_lr :: FloatingPoint
  gamma   :: FloatingPoint
end

type Inv <: LearningRatePolicy 
  base_lr :: FloatingPoint
  gamma   :: FloatingPoint
  power   :: FloatingPoint
end

type Staged <: LearningRatePolicy
  stages     :: Vector{(Int, LearningRatePolicy)}
  curr_stage :: Int
  iter_base  :: Int
  Staged(stages...) = begin
    new([(n,convert(LearningRatePolicy,p)) for (n,p) in stages], 1, 0)
  end
end

end # module LRPolicy

get_learning_rate(policy::LRPolicy.Fixed, state::SolverState) = policy.base_lr
get_learning_rate(policy::LRPolicy.Step, state::SolverState) =
    policy.base_lr * policy.gamma ^ (floor(state.iter / policy.stepsize))
get_learning_rate(policy::LRPolicy.Exp, state::SolverState) =
    policy.base_lr * policy.gamma ^ state.iter
get_learning_rate(policy::LRPolicy.Inv, state::SolverState) =
    policy.base_lr * (1 + policy.gamma * state.iter) ^ (-policy.power)

function get_learning_rate(policy::LRPolicy.Staged, state::SolverState)
  maxiter = policy.stages[policy.curr_stage][1]
  if maxiter <= 0 || policy.curr_stage == length(policy.stages)
    # stay in this stage forever if
    #  - maxiter is set to 0
    #  - this is already the last stage
    return get_learning_rate(policy.stages[policy.curr_stage][2], state)
  end

  iter = state.iter - policy.iter_base
  if iter >= maxiter
    policy.iter_base = iter
    policy.curr_stage += 1
  end
  return get_learning_rate(policy.stages[policy.curr_stage][2], state)
end

@defstruct SolverParameters Any (
  lr_policy :: LearningRatePolicy = LRPolicy.Fixed(0.01),
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
function check_coffee_breaks(t::CoffeeBreakTimeType, solver::Solver, state::SolverState, net::Net)
  for cb in solver.coffee_breaks
    check_coffee_break(cb, t, state, net)
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
  if state.iter >= solver.params.max_iter
    return true
  end
  return false
end

include("solvers/sgd.jl")
