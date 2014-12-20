include("gradient-checking.jl")

function test_simple_net_gradient(backend)
	println("-- Testing gradients on simple network (example for gradient checking code)")

	batch_size = 1
	X = randn(10, batch_size)
	Y = ones(1, batch_size)
	data = MemoryDataLayer(name="data", data=Array[X], tops=[:data], batch_size=batch_size)
	label = MemoryDataLayer(name="label", data=Array[Y], tops=[:label], batch_size=batch_size)
	fc1_layer = InnerProductLayer(name="ip1", output_dim=2, neuron=Neurons.ReLU(), bottoms=[:data], tops=[:ip1])
	loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:ip1, :label])
	net = Net("simple", backend, [data, label, fc1_layer, loss_layer])

	copy!(net.states[3].parameters[1].blob, randn(10*2))
	copy!(net.states[3].parameters[2].blob, randn(2))

	test_gradients(net, 1e-4)
end

if test_cpu
	test_simple_net_gradient(backend_cpu)
end
