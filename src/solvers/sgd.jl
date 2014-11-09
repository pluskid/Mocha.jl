type SGD <: Solver
end

function solve(sgd::SGD, net::Net)
  param_states = filter(x -> :parameters ∈ names(x), net.states)

  param_history = Array(Vector{Blob}, length(param_states))
  for i = 1:length(param_states)
    state = param_states[i]
    param_history[i] = [make_zero_blob(net.sys.backend, eltype(x.blob),size(x.blob)...) for x in state.parameters]
  end

  solver_state = init(net)
  while true
    forward_backward(solver_state, net)

    # update parameters
    for i = 1:length(param_states)
      state = param_states[i]
      history = param_history[i]
      for j = 1:length(state.parameters)
        blob = history[j]
        gradient = state.parameters[j].gradient
        data_type = eltype(blob)

        #tmp = zeros(eltype(gradient), size(state.parameters[j].blob))
        #copy!(tmp, gradient)
        #println("--gradient: $(maximum(abs(tmp)))")
        #copy!(tmp, state.parameters[j].blob)
        #println("--before update: $(maximum(abs(tmp)))")
        update_parameters(net, state, state.parameters[j].blob, blob, gradient, data_type)
        #copy!(tmp, state.parameters[j].blob)
        #println("--after update: $(maximum(abs(tmp)))")
      end
    end

    if stop_condition_satisfied(solver_state, net)
      break
    end
  end
end

function update_parameters(net::Net{CPUBackend}, state, param_blob, blob, gradient, data_type)
  # blob = net.sys.momentum * blob
  BLAS.scal!(length(blob), convert(data_type, net.sys.momentum), blob.data, 1)
  # blob = - net.sys.learning_rate * gradient + blob
  BLAS.axpy!(length(blob), convert(data_type, -net.sys.learning_rate), gradient.data, 1, blob.data, 1)

  # update parameter
  # param_blob += blob
  BLAS.axpy!(length(blob), convert(data_type, 1), blob.data, 1, param_blob.data, 1)
end
function update_parameters(net::Net{CuDNNBackend}, state, param_blob, blob, gradient, data_type)
  # blob = net.sys.momentum * blob
  CuBLAS.scal(net.sys.backend.cublas_ctx, length(blob), convert(data_type, net.sys.momentum),
      blob.ptr, 1)
  # blob = - net.sys.learning_rate * gradient + blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, -net.sys.learning_rate),
      gradient.ptr, 1, blob.ptr, 1)

  # update parameter
  # param_blob += blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, 1),
      blob.ptr, 1, param_blob.ptr, 1)
end
