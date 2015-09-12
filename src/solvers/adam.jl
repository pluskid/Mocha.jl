export Adam

immutable Adam <: SolverMethod
end

function make_solver_parameters(solver::Adam; kwargs...)
  p = make_solver_parameters(lr_policy=LRPolicy.Inv(0.003, 0.0001, 0.5),
                             beta1=0.9,
                             beta2=0.999,
                             epsilon=1e-8)
  merge(p, SolverParameters(kwargs))
end


validate_parameters(solver::Adam, params::SolverParameters) = begin
    validate_parameters(params, :lr_policy, :beta1, :beta2, :epsilon)
end

type AdamSolverState <: InternalSolverState
  param_states        :: Vector{LayerState}
  grad_1st_moment_est :: Vector{Vector{Blob}} # Exponentially weighted moving average - biased estimate of 1st moment of gradient
  grad_2nd_moment_est :: Vector{Vector{Blob}} # Exponentially weighted moving average - biased estimate of raw 2nd moment of gradient
  t                   :: Float64  # timestep since estimates initialized
  learning_rate       :: Float64
end

type AdamSolverStateSnapshot <: SolverStateSnapshot
    iter                :: Int
    obj_val             :: Float64
    grad_1st_moment_est :: Vector{Vector{Array}}
    grad_2nd_moment_est :: Vector{Vector{Array}}
    t                   :: Float64  # timestep since estimates initialized
    learning_rate       :: Float64
end

function blobs_clone(blobs::Vector{Vector{Blob}})
    out = Array(Vector{Array}, length(blobs))
    for (i, vecblobs) in enumerate(blobs)
        out[i] = [Array(eltype(b), size(b)) for b in vecblobs]
        for (dst, b) in zip(out[i], vecblobs)
            copy!(dst, b)
        end
    end
    return out
end

snapshot(state::SolverState{AdamSolverState}) = begin
    AdamSolverStateSnapshot(state.iter, state.obj_val,
                            blobs_clone(state.internal.grad_1st_moment_est),
                            blobs_clone(state.internal.grad_2nd_moment_est),
                            state.internal.t,
                            state.internal.learning_rate)
end

function solver_state(net::Net, snapshot::AdamSolverStateSnapshot)
    i_state = AdamSolverState(snapshot.learning_rate, net)
    for i in 1:length(i_state.param_states)
        for (dst, src) in zip(i_state.grad_1st_moment_est[i],
                              snapshot.grad_1st_moment_est[i])
            copy!(dst, src)
        end
        for (dst, src) in zip(i_state.grad_2nd_moment_est[i],
                              snapshot.grad_2nd_moment_est[i])
            copy!(dst, src)
        end
    end
    i_state.t = snapshot.t
    SolverState(snapshot.iter, snapshot.obj_val, Dict(), i_state)
end

function solver_state(method::Adam, net::Net, params::SolverParameters)
    learning_rate = get_learning_rate(params[:lr_policy])
    SolverState(AdamSolverState(learning_rate, net))
end

AdamSolverState(learning_rate::Float64, net::Net) = begin
    param_states = updatable_layer_states(net)

    grad_1st_moment_est = Array(Vector{Blob}, length(param_states))
    grad_2nd_moment_est = Array(Vector{Blob}, length(param_states))

    for i = 1:length(param_states)
        layerstate = param_states[i]
        grad_1st_moment_est[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in layerstate.parameters]
        grad_2nd_moment_est[i] = [make_zero_blob(net.backend, eltype(x.blob),size(x.blob)...) for x in layerstate.parameters]
    end

    return AdamSolverState(param_states, grad_1st_moment_est, grad_2nd_moment_est, 0, learning_rate)
end


function shutdown(state::SolverState{AdamSolverState})
  map(x -> map(destroy, x), state.internal.grad_1st_moment_est)
  map(x -> map(destroy, x), state.internal.grad_2nd_moment_est)
end


function update(solver::Solver{Adam}, net::Net, state::SolverState{AdamSolverState})
  state.internal.t += 1
  state.internal.learning_rate = get_learning_rate(solver.params[:lr_policy], state)
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
                         state.internal.learning_rate,
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
