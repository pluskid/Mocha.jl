function test_tanh_neuron(backend::Backend, T, eps)
  println("-- Testing Tanh neuron on $(typeof(backend)){$T}...")

  data = rand(T, 3,4,5,6) - 0.5
  data_blob = make_blob(backend, data)
  neuron = Neurons.Tanh()

  println("    > Forward")
  forward(backend, neuron, data_blob)
  expected_data = (1 - exp(-2data)) ./ (1 + exp(-2data))
  got_data = zeros(T, size(data))
  copy!(got_data, data_blob)

  @test all(-eps .< got_data - expected_data .< eps)

  println("    > Backward")
  grad = rand(T, size(data))
  grad_blob = make_blob(backend, grad)
  backward(backend, neuron, data_blob, grad_blob)

  expected_grad = grad .* (1 - (expected_data .* expected_data))
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, grad_blob)
  @test all(-eps .< got_grad - expected_grad .< eps)
end
function test_tanh_neuron(backend::Backend)
  test_tanh_neuron(backend, Float32, 1e-3)
  test_tanh_neuron(backend, Float64, 1e-9)
end

if test_cpu
  test_tanh_neuron(backend_cpu)
end
if test_gpu
  test_tanh_neuron(backend_gpu)
end

