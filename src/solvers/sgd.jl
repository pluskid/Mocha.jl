export SGD

immutable SGD <: SolverMethod
end

validate_parameters(method::SGD, params::SolverParameters) = validate_parameters(params, :lr_policy, :mom_policy)
make_solver_parameters(method::SGD; kwargs...) = merge(make_solver_parameters(),
                                                       [:lr_policy => LRPolicy.Fixed(0.01),
                                                        :mom_policy => MomPolicy.Fixed(0.),
                                                        ],
                                                       SolverParameters(kwargs))

type SGDSolverState <: InternalSolverState
    learning_rate :: Float64
    momentum      :: Float64
    param_states  :: Vector{LayerState}
    param_history :: Vector{Vector{Blob}}
    last_momentum :: Float64
end

type SGDSolverSnapshot <: SolverStateSnapshot
    iteration     :: Int
    obj_val       :: Float64
    learning_rate :: Float64
    momentum      :: Float64
end


############################################################
#  API functions to be implemented by each solver instance
############################################################

list_statistics(method::SGD) = ["obj_val", "iter", "learning_rate", "momentum"]

function snapshot(state::SolverState{SGDSolverState})
    SGDSolverSnapshot(state.iter, state.obj_val,
                      state.internal.learning_rate, state.internal.momentum)
end

solver_state(net::Net, snapshot::SGDSolverSnapshot) = begin
    SolverState{SGDSolverState}(snapshot.iteration,
                                snapshot.obj_val,
                                SGDSolverState(net, snapshot.learning_rate, snapshot.momentum))
end

solver_state(method::SGD, net::Net, params::SolverParameters) = begin
    SolverState(SGDSolverState(net,
                               get_learning_rate(params[:lr_policy]),
                               get_momentum(params[:mom_policy])))
end


SGDSolverState(net::Net, learning_rate::Float64, momentum::Float64) = begin
  param_states = updatable_layer_states(net)

  param_history = Array(Vector{Blob}, length(param_states))

  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end
a
  return SGDSolverState(learning_rate, momentum, param_states, param_history, momentum)
end

function shutdown(state::SolverState{SGDSolverState})
  map(x -> map(destroy, x), state.internal.param_history)
end

function update(solver::Solver{SGD}, net::Net, state::SolverState{SGDSolverState})
  for i = 1:length(state.internal.param_states)
    layer_state   = state.internal.param_states[i]
    history = state.internal.param_history[i]
    for j = 1:length(layer_state.parameters)
      hist_blob = history[j]
      gradient  = layer_state.parameters[j].gradient
      data_type = eltype(hist_blob)

      # to keep iteration <-> momentum correspondence consistent with the Nesterov solver,
      # we use the momentum from the last iteration
      update_parameters!(net, solver.method, layer_state.parameters[j].learning_rate * state.internal.learning_rate,
          state.internal.last_momentum, layer_state.parameters[j].blob, hist_blob, gradient, data_type)
    end
  end
  state.internal.last_momentum = state.internal.momentum
end


function update_parameters!(net::Net{CPUBackend}, method::SGD, learning_rate, momentum, param_blob, hist_blob, gradient, data_type)
  # hist_blob = momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, momentum), pointer(hist_blob.data), 1)
  # hist_blob = -learning_rate * gradient + hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, -learning_rate), pointer(gradient.data), 1, pointer(hist_blob.data), 1)

  # update parameter
  # param_blob += hist_blob
  BLAS.axpy!(length(hist_blob), convert(data_type, 1), pointer(hist_blob.data), 1, pointer(param_blob.data), 1)
end
