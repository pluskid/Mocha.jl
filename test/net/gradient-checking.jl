function compute_fd_and_grad_change(net::Net, epsilon::Float64)
	## first, calculate the change of the function value through finite differences

	# let theta be the initial parameter, then we evaluate the network at theta and theta + 2*epsilon;
	# the gradient will be evaluated theta + epsilon; this gives a symmetric finite difference approximation 

	# evaluate at theta
	forward(net)
	diff = net.states[end].loss

	L = length(net.layers)
	perturbation = Array(Any, L)
	for l = 1:L
		if Mocha.has_param(net.layers[l])
			M = length(net.states[l].parameters)
			perturbation[l] = Array(Any, M)
			for m = 1:M
				perturbation[l][m] = randn(size(net.states[l].parameters[m].blob.data))
			end
		end
	end

	# move from theta to theta + 2*epsilon
	for l = 1:L
		if Mocha.has_param(net.layers[l])
			for m = 1:length(net.states[l].parameters)
				BLAS.axpy!(2*epsilon, perturbation[l][m], net.states[l].parameters[m].blob.data)
			end
		end
	end

	# evaluate at theta + 2*epsilon
	forward(net)
	diff = (net.states[end].loss - diff)/(2*epsilon) # finite difference approximation

	# move back to theta + epsilon
	for l = 1:L
		if Mocha.has_param(net.layers[l])
			for m = 1:length(net.states[l].parameters)
				BLAS.axpy!(-epsilon, perturbation[l][m], net.states[l].parameters[m].blob.data)
			end
		end
	end

	## second, calculate the change of the function value by using the gradient
	forward(net)
	backward(net)
	grad_diff = 0.0
	for l in 1:L
		if Mocha.has_param(net.layers[l])
			M = length(net.states[l].parameters)
			for m in 1:M
				grad_diff += dot(net.states[l].parameters[m].gradient.data[:], perturbation[l][m][:])
			end
		end
	end

	return (diff, grad_diff)
end

function test_gradients(net::Net, epsilon::Float64, repetitions=10)
	(diff, grad_diff) = compute_fd_and_grad_change(net, epsilon)
	for i = 1:repetitions
		@test abs(diff - grad_diff) <= epsilon
	end
end