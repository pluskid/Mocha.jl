export SolverParameters
export SGD

export LearningRatePolicy, LRPolicy, get_learning_rate, MomentumPolicy, MomPolicy, get_momentum

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

  Staged(stages...) = begin
    accum_stages = Array((Int, LearningRatePolicy), length(stages))
    accum_iter = 0
    for i = 1:length(stages)
      (n, lrp) = stages[i]
      accum_iter += n
      accum_stages[i] = (accum_iter, convert(LearningRatePolicy, lrp))
    end

    new(accum_stages, 1)
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
  if policy.curr_stage == length(policy.stages)
    # already in the last stage, stick there forever
  else
    maxiter = policy.stages[policy.curr_stage][1]
    while state.iter >= maxiter && policy.curr_stage < length(policy.stages)
      policy.curr_stage += 1
      @info("Staged learning rate policy: switching to stage $(policy.curr_stage)")
      maxiter = policy.stages[policy.curr_stage][1]
    end
  end
  return get_learning_rate(policy.stages[policy.curr_stage][2], state)
end


############################################################
# Momentum policy
############################################################
abstract MomentumPolicy
module MomPolicy
using ..Mocha.MomentumPolicy
type Fixed <: MomentumPolicy
  base_mom :: FloatingPoint
end

# min(base_mom * gamma ^ (floor(iter / stepsize)), max_mom)
type Step <: MomentumPolicy
  base_mom :: FloatingPoint
  gamma    :: FloatingPoint
  stepsize :: Int
  max_mom  :: FloatingPoint
end

type Linear <: MomentumPolicy
  base_mom :: FloatingPoint
  gamma    :: FloatingPoint
  stepsize :: Int
  max_mom  :: FloatingPoint
end

end # module MomPolicy

get_momentum(policy::MomPolicy.Fixed, state::SolverState) = policy.base_mom
get_momentum(policy::MomPolicy.Step, state::SolverState) =
    min(policy.base_mom * policy.gamma ^ (floor(state.iter / policy.stepsize)), policy.max_mom)
get_momentum(policy::MomPolicy.Linear, state::SolverState) =
    min(policy.base_mom + floor(state.iter / policy.stepsize) * policy.gamma, policy.max_mom)

@defstruct SolverParameters Any (
  lr_policy :: LearningRatePolicy = LRPolicy.Fixed(0.01),
  mom_policy  :: MomentumPolicy = MomPolicy.Fixed(0.),
  (max_iter :: Int = 0, max_iter > 0),
  (regu_coef :: FloatingPoint = 0.0005, regu_coef >= 0),
)

############################################################
# Coffee break utilities
############################################################
#-- This function is to be called by the end-user
function setup_coffee_lounge(solver::Solver; save_into::String="", every_n_iter::Int=1, file_exists=:merge)
  solver.coffee_lounge = CoffeeLounge(filename=save_into, save_every_n_iter=every_n_iter, file_exists=file_exists)
end

function add_coffee_break(solver::Solver, coffee::Coffee; every_n_iter::Int=0, every_n_epoch::Int=0)
  cb = CoffeeBreak(coffee, every_n_iter, every_n_epoch)
  push!(solver.coffee_breaks, cb)
end

############################################################
# General utilities that could be used by all solvers
############################################################
function update_solver_state(state::SolverState, obj_val :: Float64)
  state.obj_val = obj_val
end
function update_solver_time(state::SolverState)
  state.iter += 1
end
function stop_condition_satisfied(solver::Solver, state::SolverState, net::Net)
  if state.iter >= solver.params.max_iter
    return true
  end
  return false
end

include("solvers/sgd.jl")
