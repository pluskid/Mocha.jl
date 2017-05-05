# An implementation of Adadelta: An Adaptive Learning Rate Method 
# in Mocha.jl
# CREATED BY: ALEXANDER AMINI
#################################################################

export Adadelta

immutable Adadelta <: SolverMethod
end

make_solver_parameters(method::Adadelta; kwargs...) =
  merge( make_solver_parameters(rho=0.95, eps=1e-6), SolverParameters(kwargs) )

validate_parameters(method::Adadelta, params::SolverParameters) = validate_parameters(params, :rho, :eps)

type AdadeltaSolverState <: InternalSolverState
    param_states  :: Vector{LayerState}
    gradients_sq :: Vector{Vector{Blob}}
    deltas_sq :: Vector{Vector{Blob}}
end

type AdadeltaSolverSnapshot <: SolverStateSnapshot
    iteration     :: Int
    obj_val       :: Float64
end

function snapshot(state::SolverState{AdadeltaSolverState})
    AdadeltaSolverSnapshot(state.iter, state.obj_val)
end

solver_state(net::Net, snapshot::AdadeltaSolverSnapshot) = begin
    SolverState{AdadeltaSolverState}(snapshot.iteration, snapshot.obj_val,
                                Dict(), AdadeltaSolverState(net))
end

solver_state(method::Adadelta, net::Net, params::SolverParameters) = begin
    SolverState(AdadeltaSolverState(net))
end


AdadeltaSolverState(net::Net) = begin
  param_states = updatable_layer_states(net)

  gradients_sq = Array{Vector{Blob}}(length(param_states))
  deltas_sq = Array{Vector{Blob}}(length(param_states))

  for i = 1:length(param_states)
    state = param_states[i]
    gradients_sq[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
    deltas_sq[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end
  return AdadeltaSolverState(param_states, gradients_sq, deltas_sq)
end


function shutdown(state::SolverState{AdadeltaSolverState})
  map(x -> map(destroy, x), state.internal.gradients_sq)
  map(x -> map(destroy, x), state.internal.deltas_sq)
end

function update(solver::Solver{Adadelta}, net::Net, state::SolverState{AdadeltaSolverState})

  for i = 1:length(state.internal.param_states)
    layer_state   = state.internal.param_states[i]
    gradients_sq = state.internal.gradients_sq[i]
    deltas_sq = state.internal.deltas_sq[i]
    for j = 1:length(layer_state.parameters)
      gradSq = gradients_sq[j]
      deltaSq = deltas_sq[j]
      gradient  = layer_state.parameters[j].gradient
      data_type = eltype(gradSq)

      update_parameters!(net, solver.method, solver.params[:rho], solver.params[:eps],
          layer_state.parameters[j].blob, gradSq, deltaSq, gradient, data_type)
    end
  end
end

function update_parameters!(net::Net{CPUBackend}, method::Adadelta, rho, eps,
    param_blob, gradSq, deltaSq, gradient, data_type)  

  BLAS.scal!(length(gradSq), convert(data_type, rho), gradSq.data, 1)
  BLAS.axpy!(length(gradSq), convert(data_type, 1-rho), pointer(gradient.data.^2), 1, gradSq.data, 1)

  deltas = (sqrt(deltaSq.data+eps) ./ sqrt(gradSq.data+eps)) .* gradient.data
  BLAS.scal!(length(gradSq), convert(data_type, rho), deltaSq.data, 1)
  BLAS.axpy!(length(gradSq), convert(data_type, 1-rho), pointer(deltas.^2), 1, deltaSq.data, 1)

  BLAS.axpy!(length(gradSq), convert(data_type, -1), pointer(deltas), 1, param_blob.data, 1)
  
end
