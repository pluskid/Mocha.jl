# An implementation of Nesterov's Accelerated method based on a reformulation
# that makes it similar to ordinary (Stochastic) Gradient Descent with Momentum
#
# Yoshua Bengio, Nicolas Boulanger-Lewandowski, Razvan Pascanu. Advances in
# Optimizing Recurrent Networks. arXiv:1212.0901 [cs.LG]


immutable Nesterov <: SolverMethod
end
type NesterovSolverState <: InternalSolverState
    learning_rate :: Float64
    momentum      :: Float64
    param_states  :: Vector{LayerState}
    param_history :: Vector{Vector{Blob}}
    last_momentum :: Float64
end

type NesterovSolverSnapshot <: SolverStateSnapshot
    iteration     :: Int
    obj_val       :: Float64
    learning_rate :: Float64
    momentum      :: Float64
end

#typealias NesterovSolverState SGDSolverState
#typealias NesterovSolverSnapashot SGDSolverSnapshot


function update(solver::Solver{Nesterov}, net::Net, state::SolverState{NesterovSolverState})
  # TODO check if this is identical with sgd.jl update()?
  for i = 1:length(state.internal.param_states)
    layer_state   = state.internal.param_states[i]
    history = state.internal.param_history[i]
    for j = 1:length(layer_state.parameters)
      hist_blob = history[j]
      gradient  = layer_state.parameters[j].gradient
      data_type = eltype(hist_blob)

      update_parameters!(net, solver.method, layer_state.parameters[j].learning_rate * state.internal.learning_rate,
          state.internal.last_momentum, state.internal.momentum, layer_state.parameters[j].blob, hist_blob, gradient, data_type)
    end
  end

  state.internal.last_momentum = state.internal.momentum
end


function shutdown(state::NesterovSolverState)
  map(x -> map(destroy, x), state.param_history)
end

function update_parameters!(net::Net{CPUBackend}, method::Nesterov, learning_rate,
    last_momentum, momentum, param_blob, hist_blob, gradient, data_type)

  # param_blob += -last_momentum* hist_blob (update with vt-1)
  BLAS.axpy!(length(hist_blob), convert(data_type, -last_momentum), hist_blob.data, 1, param_blob.data, 1)

  # hist_blob = last_momentum * hist_blob
  BLAS.scal!(length(hist_blob), convert(data_type, last_momentum), hist_blob.data, 1)

  # hist_blob = -learning_rate * gradient + hist_blob (calc vt)
  BLAS.axpy!(length(hist_blob), convert(data_type, -learning_rate), gradient.data, 1, hist_blob.data, 1)


  # param_blob += (1+momentum) * hist_blob (update with vt)
  BLAS.axpy!(length(hist_blob), convert(data_type, 1 + momentum), hist_blob.data, 1, param_blob.data, 1)
end
