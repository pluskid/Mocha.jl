function test_relu_neuron(sys::System, T)
  println("-- Testing ReLU neuron on $(typeof(sys.backend)){$T}...")

  eps = 1e-10
  data = rand(T, 3,4,5,6) - 0.5
  data_blob = make_blob(sys.backend, data)
  neuron = Neurons.ReLU()

  println("    > Forward")
  forward(sys, neuron, data_blob)
  expected_data = max(data, 0)
  got_data = zeros(T, size(data))
  copy!(got_data, data_blob)

  @test all(-eps .< got_data - expected_data .< eps)

  println("    > Backward")
  grad = rand(T, size(data))
  grad_blob = make_blob(sys.backend, grad)
  backward(sys, neuron, data_blob, grad_blob)

  expected_grad = grad .* (data .> 0)
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, grad_blob)

  @test all(-eps .< got_grad - expected_grad .< eps)
end
function test_relu_neuron(sys::System)
  test_relu_neuron(sys, Float32)
  test_relu_neuron(sys, Float64)
end

if test_cpu
  test_relu_neuron(sys_cpu)
end
if test_cudnn
  test_relu_neuron(sys_cudnn)
end


