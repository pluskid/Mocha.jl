export SolverParameters
export SGD, Nesterov

export LearningRatePolicy, LRPolicy, get_learning_rate, MomentumPolicy, MomPolicy, get_momentum

export setup_coffee_lounge, add_coffee_break, solve

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
  load_from :: String = ""
)

############################################################
# Coffee break utilities
############################################################
#-- This function is to be called by the end-user
function setup_coffee_lounge(solver::Solver; save_into::String="", every_n_iter::Int=1, file_exists=:merge)
  solver.coffee_lounge.filename=save_into
  solver.coffee_lounge.save_every_n_iter=every_n_iter
  solver.coffee_lounge.file_exists=file_exists
end

function add_coffee_break(solver::Solver, coffee::Coffee; kw...)
  add_coffee_break(solver.coffee_lounge, coffee; kw...)
end

############################################################
# General utilities that could be used by all solvers
############################################################
function load_snapshot(net::Net, state::SolverState, path::String)
  if isempty(path)
    return state
  end

  if endswith(path, ".hdf5") || endswith(path, ".h5")
    # load from HDF5 file, possibly exported from caffe, but training
    # from the beginning (iteration 0) as the solver state is not saved
    # in a HDF5 file
    if isfile(path)
      @info("Loading existing model from $path")
      h5open(path) do file
        load_network(file, net)
      end
    end
    return state
  else
    if endswith(path, ".jld")
      # load from some specific JLD sanpshot, the solver state is also
      # recovered
      filename = path
    else
      # automatically load from the latest snapshot in a directory
      filename = ""
      if isdir(path)
        # load the latest snapshot from the directory
        snapshots = glob(path, r"^snapshot-[0-9]+\.jld", sort_by=:mtime)
        if length(snapshots) > 0
          filename = joinpath(path, snapshots[end])
        end
      end
    end

    if !isempty(filename) && isfile(filename)
      @info("Loading existing model from $filename")
      jldopen(filename) do file
        load_network(file, net)
        return read(file, SOLVER_STATE_KEY)
      end
    else
      return state
    end
  end
end

function stop_condition_satisfied(solver::Solver, state::SolverState, net::Net)
  # state.iter counts how many iteration we have computed.
  if state.iter >= solver.params.max_iter
    return true
  end
  return false
end

############################################################
# Solver API
############################################################
abstract SolverInternelState

function setup(solver::Solver, net::Net, state::SolverState)
  error("Not implemented, should return a SolverInternelState")
end
function update(solver::Solver, net::Net, i_state::SolverInternelState, state::SolverState)
  error("Not implemented, should do one iteration of update")
end
function shutdown(solver::Solver, i_state::SolverInternelState)
  error("Not implemented, should shutdown the solver")
end

############################################################
# General Solver Loop
############################################################
function solve(solver::Solver, net::Net)
  @debug("Checking network topology for back-propagation")
  check_bp_topology(net)

  solver_state = SolverState()
  solver_state = load_snapshot(net, solver_state, solver.params.load_from)
  solver_state.learning_rate = get_learning_rate(solver.params.lr_policy, solver_state)
  solver_state.momentum = get_momentum(solver.params.mom_policy, solver_state)

  # we init network AFTER loading. If the parameters are loaded from file, the
  # initializers will be automatically set to NullInitializer
  init(net)

  # Initial forward iteration
  solver_state.obj_val = forward(net, solver.params.regu_coef)

  @debug("Initializing coffee breaks")
  setup(solver.coffee_lounge, solver_state, net)

  # coffee break for iteration 0, before everything starts
  check_coffee_break(solver.coffee_lounge, solver_state, net)

  i_state = setup(solver, net, solver_state)

  @debug("Entering solver loop")
  trainable_layers = filter(i -> has_param(net.layers[i]) && !is_frozen(net.states[i]), 1:length(net.layers))
  while true
    solver_state.iter += 1

    backward(net, solver.params.regu_coef)
    solver_state.learning_rate = get_learning_rate(solver.params.lr_policy, solver_state)
    solver_state.momentum = get_momentum(solver.params.mom_policy, solver_state)

    update(solver, net, i_state, solver_state)

    # apply weight constraints
    for i in trainable_layers
      for param in net.states[i].parameters
        cons_every = param.constraint.every_n_iter
        if cons_every > 0 && solver_state.iter % cons_every == 0
          constrain!(net.backend, param.constraint, param.blob)
        end
      end
    end

    solver_state.obj_val = forward(net, solver.params.regu_coef)
    check_coffee_break(solver.coffee_lounge, solver_state, net)

    if stop_condition_satisfied(solver, solver_state, net)
      break
    end
  end

  shutdown(solver.coffee_lounge, net)
  shutdown(solver, i_state)
end

############################################################
# Specific Solvers
############################################################
include("solvers/sgd.jl")
include("solvers/nesterov.jl")
