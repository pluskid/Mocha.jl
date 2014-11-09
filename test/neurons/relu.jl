function test_relu_neuron(sys::System)
  println("-- Testing ReLU neuron on $(typeof(sys.backend))...")

  eps = 1e-10
  data = rand(3,4,5,6) - 0.5
  data_blob = make_blob(sys.backend, data)
  neuron = Neurons.ReLU()

  println("    > Forward")
  forward(sys, neuron, data_blob)
  expected_data = max(data, 0)
  got_data = zeros(size(data))
  copy!(got_data, data_blob)

  #diff_idx = find(abs(got_data - expected_data) .> eps)
  #println("diff-idx = $diff_idx")
  #println("$(got_data[diff_idx])")
  #println("$(expected_data[diff_idx])")

  @test all(-eps .< got_data - expected_data .< eps)

  println("    > Backward")
  grad = rand(size(data))
  grad_blob = make_blob(sys.backend, grad)
  backward(sys, neuron, data_blob, grad_blob)

  expected_grad = grad .* (expected_data .> 0)
  got_grad = zeros(size(expected_grad))
  copy!(got_grad, grad_blob)

  @test all(-eps .< got_grad - expected_grad .< eps)
end

if test_cpu
  test_relu_neuron(sys_cpu)
end
if test_cudnn
  test_relu_neuron(sys_cudnn)
end


