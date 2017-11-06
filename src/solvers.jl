export SolverParameters, SolverState, Solver
export setup_coffee_lounge, add_coffee_break, solve
export load_snapshot
export make_solver_parameters
export init_solve, onestep_solve, finalize_solve
import Base.Meta: quot

@compat abstract type SolverMethod end # An enum type to identify the solver in validation functions

@compat abstract type InternalSolverState end # All the state a solver needs to update an iteration
const SolverParameters = Dict{Symbol,Any}

immutable Solver{T<:SolverMethod}
  method        :: T
  params        :: SolverParameters
  coffee_lounge :: Any # forward declaration
end

Solver{T}(method::T, params::SolverParameters) = begin
    validate_parameters(method, params)
    Solver(method, params, CoffeeLounge())
end

type SolverState{T<:InternalSolverState}
  iter               :: Int
  obj_val            :: Float64
  losses             :: Dict
  internal           :: T
end
function copy_solver_state!{T<:InternalSolverState}(dst::SolverState{T}, src::SolverState{T})
  dst.iter = src.iter
  dst.obj_val = src.obj_val
  dst.losses = src.losses
  dst.internal = src.internal
end

SolverState{T<:InternalSolverState}(internal::T) = SolverState{T}(0, Inf, Dict(), internal)

@compat abstract type SolverStateSnapshot end # Just the serializable part of the solver state, for snapshot files


function make_solver_parameters(;kwargs...)
  localDict = @compat Dict(:max_iter => Inf,
                           :regu_coef => 0.0005,
                           :load_from => "")
  merge(localDict,
        SolverParameters(kwargs))
end

function validate_parameters(params::SolverParameters, args...)
    if params[:regu_coef] < 0
        error("regu_coef must be non-negative")
    end
    for a in args
        if !haskey(params, a)
            error("Must provide parameter $a")
        end
    end
end

############################################################
#  API functions to be implemented by each solver instance
############################################################

function get_statistic(state::SolverState, name::AbstractString)
  if haskey(state.losses, name)
    return state.losses[name]
  else
    return NaN
  end
end

function get_statistic(state::SolverState, name::Symbol)
    if name in [:obj_val, :iter]
      @eval $(quot(state)).$name
    else
        @eval $(quot(state.internal)).$name
    end
end

function format_statistic(state::SolverState, name)
    format_statistic(state, name, get_statistic(state, name))
end

function format_statistic(state::SolverState, name, value)
    if name == :iter
        @sprintf("%s=%06d", name, value)
    else
        @sprintf("%s=%.8f", name, value)
    end
end


validate_parameters(method::SolverMethod, params::SolverParameters) = begin # should turn validate method-specific parameters
    validate_parameters(params)
    error("solver_state is not implemented for the method type $(typeof(method)), and params type $(typeof(params))")
end

function snapshot(state::SolverState) # should turn a SolverState{T} into a SolverStateSnapshot instance
    error("snapshot is not implemented for the state type $(typeof(state))")
end

function solver_state(net::Net, snapshot::SolverStateSnapshot) # should use a SolverStateSnapshot instance to create a SolverState{T}
    error("solver_state is not implemented for the net type $(typeof(net)), and snapshot type $(typeof(snapshot))")
end

function solver_state(solver::SolverMethod, net::Net, params::SolverParameters) # should create SolverState{T} from SolverParameters dictionary
    error("solver_state is not implemented for the solver type $(typeof(solver)), net type $(typeof(net)), and params type $(typeof(params))")
end

function update{T}(solver::Solver{T}, net::Net, state::SolverState) # should do one iteration of update
    error("update is not implemented for the solver type $(typeof(solver)), net type $(typeof(net)), and state type $(typeof(state))")
end
function shutdown(state::SolverState) # should shutdown the solver
    error("shutdown is not implemented for the $(typeof(state)) type")
end


############################################################
# General utilities that could be used by all solvers
############################################################
function load_snapshot(net::Net, path::AbstractString="", state=nothing)
  if isempty(path)
    return state
  end

  if endswith(path, ".hdf5") || endswith(path, ".h5")
    # load from HDF5 file, possibly exported from caffe, but training
    # from the beginning (iteration 0) as the solver state is not saved
    # in a HDF5 file
    if isfile(path)
      m_info("Loading existing model from $path")
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
      m_info("Loading existing model from $filename")
      jldopen(filename) do file
        load_network(file, net)
        return solver_state(net, read(file, SOLVER_STATE_KEY))
      end
    else
      return state
    end
  end
end

function stop_condition_satisfied(solver::Solver, state::SolverState, net::Net)
  if state.iter >= solver.params[:max_iter]
    return true
  end
  return false
end

############################################################
# General Solver Loop
############################################################


function init_solve(solver::Solver, net::Net)
    m_debug("#DEBUG Checking network topology for back-propagation")
    check_bp_topology(net)

    state = solver_state(solver.method, net, solver.params)
    state = load_snapshot(net, solver.params[:load_from], state)

    # we init network AFTER loading. If the parameters are loaded from file, the
    # initializers will be automatically set to NullInitializer
    init(net)
    state.obj_val = forward(net, solver.params[:regu_coef])

    m_debug("#DEBUG Initializing coffee breaks")
    setup(solver.coffee_lounge, state, net)

    # coffee break for iteration 0, before everything starts
    check_coffee_break(solver.coffee_lounge, state, net)

    return state
end

function finalize_solve(solver::Solver, net::Net, state::SolverState)
    shutdown(solver.coffee_lounge, net)
    shutdown(state)
end

function onestep_solve(solver::Solver, net::Net, state::SolverState)
    state.iter += 1
    layer_states = updatable_layer_states(net)

    backward(net, solver.params[:regu_coef])
    update(solver, net, state)

    # apply weight constraints
    for layer_state in layer_states
      for param in layer_state.parameters
        cons_every = param.constraint.every_n_iter
        if cons_every > 0 && state.iter % cons_every == 0
          constrain!(net.backend, param.constraint, param.blob)
        end
      end
    end

    state.obj_val = forward(net, solver.params[:regu_coef])

    state.losses = Dict()
    for i = 1:length(net.layers)
      if has_loss(net.layers[i])
        state.losses[net.layers[i].name] = net.states[i].loss
      end
    end
    check_coffee_break(solver.coffee_lounge, state, net)
end

function solve(solver::Solver, net::Net)
    state = init_solve(solver, net)
    do_solve_loop(solver, net, state)
    finalize_solve(solver, net, state)
end

function do_solve_loop(solver::Solver, net::Net, state::SolverState)
  m_debug("#DEBUG Entering solver loop")
  while !stop_condition_satisfied(solver, state, net)
    onestep_solve(solver,net,state)

    if stop_condition_satisfied(solver, state, net)
      break
    end
  end
  return state
end

trainable_layers(net::Net) = filter(i -> has_param(net.layers[i]) && !is_frozen(net.states[i]),
                                    1:length(net.layers))

updatable_layer_states(net::Net) = [net.states[i] for i in trainable_layers(net)]
