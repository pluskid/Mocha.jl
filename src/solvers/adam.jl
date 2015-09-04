export Adam

immutable Adam <: SolverMethod
end

function make_solver_parameters(solver::Adam; kwargs...)
  p = make_solver_parameters(stepsize=0.001,
                             beta1=0.9,
                             beta2=0.999,
                             epsilon=1e-8)
  merge(p, SolverParameters(kwargs))
end


validate_parameters(solver::Adam, params::SolverParameters) = begin
    validate_parameters(params, :stepsize, :beta1, :beta2, :epsilon)
end

type AdamSolverState <: InternalSolverState
  param_states        :: Vector{LayerState}
  grad_1st_moment_est :: Vector{Vector{Blob}} # Exponentially weighted moving average - biased estimate of 1st moment of gradient
  grad_2nd_moment_est :: Vector{Vector{Blob}} # Exponentially weighted moving average - biased estimate of raw 2nd moment of gradient
  t                   :: Float64  # timestep since estimates initialized
end

type AdamSolverStateSnapshot <: SolverStateSnapshot
    iteration::Int
    obj_val::Float64
end

snapshot(state::SolverState{AdamSolverState}) = AdamSolverStateSnapshot(state.iter, state.obj_val)

solver_state(net::Net, snapshot::AdamSolverStateSnapshot) = begin
    solver_state(iteration, obj_val,
                 AdamSolverState(net))
end

function solver_state(method::Adam, net::Net, params::SolverParameters)
    SolverState(AdamSolverState(net))
end

AdamSolverState(net::Net) = begin
    param_states = updatable_layer_states(net)

    grad_1st_moment_est = Array(Vector{Blob}, length(param_states))
    grad_2nd_moment_est = Array(Vector{Blob}, length(param_states))

    for i = 1:length(param_states)
        layerstate = param_states[i]
        grad_1st_moment_est[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in layerstate.parameters]
        grad_2nd_moment_est[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in layerstate.parameters]
    end

    return AdamSolverState(param_states, grad_1st_moment_est, grad_2nd_moment_est, 0)
end


function shutdown(state::SolverState{AdamSolverState})
  map(x -> map(destroy, x), state.internal.grad_1st_moment_est)
  map(x -> map(destroy, x), state.internal.grad_2nd_moment_est)
end


function update(solver::Solver{Adam}, net::Net, state::SolverState{AdamSolverState})
  state.internal.t += 1
  for i = 1:length(state.internal.param_states)
    layer_state   = state.internal.param_states[i]
    m       = state.internal.grad_1st_moment_est[i]
    v       = state.internal.grad_2nd_moment_est[i]
    for j = 1:length(layer_state.parameters)
      gradient  = layer_state.parameters[j].gradient
      data_type = eltype(m[j])
      # N.B. we are ignoring the parameter-specific learning rate multipliers
      # since they ought to adapt automatically.
      update_parameters!(net, solver.method,
                         solver.params[:stepsize],
                         solver.params[:epsilon],
                         solver.params[:beta1],
                         solver.params[:beta2],
                         m[j], v[j],
                         gradient,
                         layer_state.parameters[j].blob,
                         state.internal.t, data_type)
    end
  end

end

function update_parameters!(net::Net{CPUBackend}, method::Adam,
                            alpha::Float64, epsilon::Float64, beta1::Float64, beta2::Float64,
                            m, v, gradient, param_blob, t, data_type)

  # update biased gradient moment estimates, m and v
  # m_t <- beta1*m_{t-1} + (1-beta1)*g_t

  BLAS.scal!(length(m), convert(data_type, beta1), pointer(m.data), 1)
  BLAS.axpy!(length(m), convert(data_type, 1-beta1), pointer(gradient.data), 1, pointer(m.data), 1)

  # now we need g2 = g.^2
  # Are we better to preallocate (heavy memory usage) or allocate each time (slower?)
  g2 = gradient.data .^ 2 # TODO more efficient way of making a new buffer, copying and squaring?

  BLAS.scal!(length(v), convert(data_type, beta2), pointer(g2), 1)
  BLAS.axpy!(length(v), convert(data_type, 1-beta2), pointer(g2), 1, pointer(v.data), 1)

  # Correct bias and calculate effective stepsize for timestep t
  alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
  # Update param <- param - alpha_t * m/(sqrt(v) + epsilon)
  delta = m.data ./ (sqrt(v.data) .+ epsilon)
  BLAS.axpy!(length(param_blob), convert(data_type, -alpha_t), pointer(delta), 1, pointer(param_blob.data), 1)

end
