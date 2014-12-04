function test_relu_neuron(backend::Backend, T)
  println("-- Testing ReLU neuron on $(typeof(backend)){$T}...")

  eps = 1e-10
  data = rand(T, 3,4,5,6) - 0.5
  data_blob = make_blob(backend, data)
  neuron = Neurons.ReLU()

  println("    > Forward")
  forward(backend, neuron, data_blob)
  expected_data = max(data, 0)
  got_data = zeros(T, size(data))
  copy!(got_data, data_blob)

  @test all(-eps .< got_data - expected_data .< eps)

  println("    > Backward")
  grad = rand(T, size(data))
  grad_blob = make_blob(backend, grad)
  backward(backend, neuron, data_blob, grad_blob)

  expected_grad = grad .* (data .> 0)
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, grad_blob)

  @test all(-eps .< got_grad - expected_grad .< eps)
end
function test_relu_neuron(backend::Backend)
  test_relu_neuron(backend, Float32)
  test_relu_neuron(backend, Float64)
end

if test_cpu
  test_relu_neuron(backend_cpu)
end
if test_cudnn
  test_relu_neuron(backend_cudnn)
end


