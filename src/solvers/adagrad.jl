####################################################################
#### An implementation of Adagrad: Adaptive Subgradient Methods ####
###########for Online Learning and Stochastic Optimization##########
########################### in Mocha.jl ############################
################### CREATED BY: ALEXANDER AMINI ####################
####################################################################

export Adagrad

immutable Adagrad <: SolverMethod
end

make_solver_parameters(method::Adagrad; kwargs...)=
 merge( make_solver_parameters(gamma=1.0, epsilon=1e-8), SolverParameters(kwargs))

validate_parameters(method::Adagrad, params::SolverParameters) = validate_parameters(params, :gamma, :epsilon)

type AdagradSolverState <: InternalSolverState
    param_states  :: Vector{LayerState}
    param_history :: Vector{Vector{Blob}}
end

type AdagradSolverSnapshot <: SolverStateSnapshot
    iteration     :: Int
    obj_val       :: Float64
end

function snapshot(state::SolverState{AdagradSolverState})
    AdagradSolverSnapshot(state.iter, state.obj_val)
end

solver_state(net::Net, snapshot::AdagradSolverSnapshot) = begin
    SolverState{AdagradSolverState}(snapshot.iteration, snapshot.obj_val,
                                Dict(),  AdagradSolverState(net))
end

solver_state(method::Adagrad, net::Net, params::SolverParameters) = begin
    SolverState(AdagradSolverState(net ))
end

AdagradSolverState(net::Net) = begin
  param_states = updatable_layer_states(net)

  param_history = Array{Vector{Blob}}(length(param_states))

  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end
  return AdagradSolverState(param_states, param_history)
end

function shutdown(state::SolverState{AdagradSolverState})
  map(x -> map(destroy, x), state.internal.param_history)
end

function update(solver::Solver{Adagrad}, net::Net, state::SolverState{AdagradSolverState})
  for i = 1:length(state.internal.param_states)
    layer_state   = state.internal.param_states[i]
    history = state.internal.param_history[i]
    for j = 1:length(layer_state.parameters)
      hist_blob = history[j]
      gradient  = layer_state.parameters[j].gradient
      data_type = eltype(hist_blob)

      update_parameters!(net, solver.method, solver.params[:gamma], solver.params[:epsilon],
          layer_state.parameters[j].blob, hist_blob, gradient, data_type)
    end
  end
end

function update_parameters!(net::Net{CPUBackend}, method::Adagrad, gamma, epsilon,
    param_blob, hist_blob, gradient, data_type)

  g2 = gradient.data .^ 2; 

  # hist_blob += 1* g2 (update with vt-1)
  BLAS.axpy!(length(hist_blob), convert(data_type, 1), pointer(g2), 1, pointer(hist_blob.data), 1)
  adj_learning_rate = gamma / (epsilon + sqrt(sum(hist_blob.data)))

  # param_blob += -adj_learning_rate * gradient 
  BLAS.axpy!(length(hist_blob), convert(data_type, -adj_learning_rate), gradient.data, 1, param_blob.data, 1)
end
