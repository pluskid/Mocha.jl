# Implementations of LR and Mom policies (some depend on coffee-break.jl)

export LRPolicy, get_learning_rate, MomPolicy, get_momentum

############################################################
# Learning rate policy
############################################################


module LRPolicy
using ..Mocha
using Compat
type Fixed <: LearningRatePolicy
  base_lr :: AbstractFloat
end

# base_lr * gamma ^ (floor(iter / stepsize))
type Step <: LearningRatePolicy
  base_lr  :: AbstractFloat
  gamma    :: AbstractFloat
  stepsize :: Int
end

# base_lr * gamma ^ iter
type Exp <: LearningRatePolicy
  base_lr :: AbstractFloat
  gamma   :: AbstractFloat
end

type Inv <: LearningRatePolicy
  base_lr :: AbstractFloat
  gamma   :: AbstractFloat
  power   :: AbstractFloat
end

# curr_lr *= gamma whenever performance
# drops on the validation set
function decay_on_validation_listener(policy, key::AbstractString, coffee_lounge::CoffeeLounge, net::Net, state::SolverState)
  stats = get_statistics(coffee_lounge, key)
  index = sort(collect(keys(stats)))
  if length(index) > 1
    if (policy.higher_better && stats[index[end]] < stats[index[end-1]]) ||
      (!policy.higher_better && stats[index[end]] > stats[index[end-1]])
      # performance drop
      Mocha.info(@sprintf("lr decay %e -> %e", policy.curr_lr, policy.curr_lr*policy.gamma))
      policy.curr_lr *= policy.gamma

      # revert to a previously saved "good" snapshot
      if isa(policy.solver, Solver)
        Mocha.info("reverting to previous saved snapshot")
        solver_state = load_snapshot(net, policy.solver.params[:load_from], state)
        Mocha.info("snapshot at iteration $(solver_state.iter) loaded")
        copy_solver_state!(state, solver_state)
      end
    end
  end
end

type DecayOnValidation <: LearningRatePolicy
  gamma       :: AbstractFloat

  key           :: AbstractString
  curr_lr       :: AbstractFloat
  min_lr        :: AbstractFloat
  listener      :: Function
  solver        :: Any
  initialized   :: Bool
  higher_better :: Bool # set to false if performance score is the lower the better

  DecayOnValidation(base_lr, key, gamma=0.5, min_lr=1e-8; higher_better=true) = begin
    policy = new(gamma, key, base_lr, min_lr)
    policy.solver = nothing
    policy.listener = (coffee_lounge,net,state) -> begin
      if policy.curr_lr < policy.min_lr
        # do nothing if we already fall below the minimal learning rate
        return
      end
      decay_on_validation_listener(policy, key, coffee_lounge, net, state)
    end
    policy.initialized = false
    policy.higher_better = higher_better

    policy
  end
end

using Compat
type Staged <: LearningRatePolicy
  stages     :: Vector{@compat(Tuple{Int, LearningRatePolicy})}
  curr_stage :: Int

  Staged(stages...) = begin
    accum_stages = Array(@compat(Tuple{Int, LearningRatePolicy}), length(stages))
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

get_learning_rate(policy::LearningRatePolicy) = policy.base_lr # Need an initial rate to create the state
get_learning_rate(policy::LRPolicy.Staged) = get_learning_rate(policy.stages[policy.curr_stage][2])
get_learning_rate(policy::LRPolicy.DecayOnValidation) = policy.curr_lr

get_learning_rate(policy::LRPolicy.Fixed, state::SolverState) = policy.base_lr
get_learning_rate(policy::LRPolicy.Step, state::SolverState) =
    policy.base_lr * policy.gamma ^ (floor(state.iter / policy.stepsize))
get_learning_rate(policy::LRPolicy.Exp, state::SolverState) =
    policy.base_lr * policy.gamma ^ state.iter
get_learning_rate(policy::LRPolicy.Inv, state::SolverState) =
    policy.base_lr * (1 + policy.gamma * state.iter) ^ (-policy.power)


function setup(policy::LRPolicy.DecayOnValidation, validation::ValidationPerformance, solver::Solver)
  register(validation, policy.listener)
  policy.solver = solver
end

get_learning_rate(policy::LRPolicy.DecayOnValidation, state::SolverState) = begin
    # TODO this is really for a SolverState{SGDSolverState} but that's a forward declaration
  if !policy.initialized
    if state.internal.learning_rate > 0
      # state.learning_rate is initialized to 0, if it is non-zero, then this might
      # be loaded from some saved snapshot, we try to align with that
      @info("Switching to base learning rate $(state.specific.learning_rate)")
      policy.curr_lr = state.internal.learning_rate
    end
    policy.initialized = true
  end

  policy.curr_lr
end

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

module MomPolicy
using ..Mocha.MomentumPolicy
using Compat
type Fixed <: MomentumPolicy
  base_mom :: AbstractFloat
end

# min(base_mom * gamma ^ (floor(iter / stepsize)), max_mom)
type Step <: MomentumPolicy
  base_mom :: AbstractFloat
  gamma    :: AbstractFloat
  stepsize :: Int
  max_mom  :: AbstractFloat
end

type Linear <: MomentumPolicy
  base_mom :: AbstractFloat
  gamma    :: AbstractFloat
  stepsize :: Int
  max_mom  :: AbstractFloat
end

using Compat
type Staged <: MomentumPolicy
  stages     :: Vector{@compat(Tuple{Int, MomentumPolicy})}
  curr_stage :: Int

  Staged(stages...) = begin
    accum_stages = Array(@compat(Tuple{Int, MomentumPolicy}), length(stages))
    accum_iter = 0
    for i = 1:length(stages)
      (n, mmp) = stages[i]
      accum_iter += n
      accum_stages[i] = (accum_iter, convert(MomentumPolicy, mmp))
    end

    new(accum_stages, 1)
  end
end

end # module MomPolicy

get_momentum(policy::MomentumPolicy) = policy.base_mom # Need an initial rate to create the state
get_momentum(policy::MomPolicy.Fixed, state::SolverState) = policy.base_mom
get_momentum(policy::MomPolicy.Step, state::SolverState) =
    min(policy.base_mom * policy.gamma ^ (floor(state.iter / policy.stepsize)), policy.max_mom)
get_momentum(policy::MomPolicy.Linear, state::SolverState) =
    min(policy.base_mom + floor(state.iter / policy.stepsize) * policy.gamma, policy.max_mom)

function get_momentum(policy::MomPolicy.Staged, state::SolverState)
  if policy.curr_stage == length(policy.stages)
    # already in the last stage, stick there forever
  else
    maxiter = policy.stages[policy.curr_stage][1]
    while state.iter >= maxiter && policy.curr_stage < length(policy.stages)
      policy.curr_stage += 1
      @info("Staged momentum policy: switching to stage $(policy.curr_stage)")
      maxiter = policy.stages[policy.curr_stage][1]
    end
  end
  return get_momentum(policy.stages[policy.curr_stage][2], state)
end

####
