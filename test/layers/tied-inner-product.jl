function test_tied_inner_product_layer(backend::Backend, n_input, T, eps)
  println("-- Testing TiedInnerProductLayer on $(typeof(backend)){$T}...")

  batch_size = rand(5:14)
  orig_dim = 20
  hidden_dim = 10

  inputs = Array[rand(T, orig_dim, batch_size) for i=1:n_input]
  b1 = rand(T, hidden_dim, 1)
  b2 = rand(T, orig_dim, 1)
  param_key = "shared-ip"

  println("    > Setup")
  registry_reset(backend)
  layer_data = MemoryDataLayer(tops=[Symbol("i-$i") for i = 1:n_input],
      batch_size=batch_size, data=inputs)
  layer_ip = InnerProductLayer(name="ip1", param_key=param_key,
      tops=[Symbol("ip1-$i") for i=1:n_input], bottoms=[Symbol("i-$i") for i=1:n_input],
      output_dim=hidden_dim)
  layer_ip2 = TiedInnerProductLayer(name="ip2", tied_param_key=param_key,
      tops=[Symbol("ip2-$i") for i=1:n_input], bottoms=[Symbol("ip1-$i") for i=1:n_input])

  net = Net("test-tied-ip", backend, [layer_data,layer_ip,layer_ip2])
  init(net)
  copy!(net.states[2].b, b1)
  copy!(net.states[3].b, b2)
  W = to_array(net.states[2].W)

  println("    > Forward")
  forward(net)
  for i = 1:n_input
    expected_output = W * (W' * inputs[i] .+ b1) .+ b2
    got_output = to_array(net.states[3].blobs[i])
    @test all(abs.(expected_output - got_output) .< eps)
  end

  println("    > Backward")
  # This network is invalid for back-propagation because there is
  # no top layer that compute loss
  @test_throws TopologyError check_bp_topology(net)

  # however, we will manually set the diffs for the top-most layer
  # to do a fake back-propagate test
  top_diffs = Array[rand(T, orig_dim, batch_size) for i=1:n_input]
  for i = 1:n_input
    copy!(net.states[3].blobs_diff[i], top_diffs[i])
  end
  backward(net)

  bias_grad = to_array(net.states[3].∇b)
  bias_grad_expected = sum([sum(top_diffs[i],2) for i = 1:n_input])
  @test all(abs.(bias_grad - bias_grad_expected) .< eps)

  for i = 1:n_input
    back_grad = to_array(net.states[2].blobs_diff[i])
    back_grad_expected = W' * top_diffs[i]
    @test all(abs.(back_grad - back_grad_expected) .< eps)
  end

  destroy(net)
end

function test_tied_inner_product_layer(backend::Backend)
  test_tied_inner_product_layer(backend, 3, Float32, 1e-3)
  test_tied_inner_product_layer(backend, 3, Float64, 1e-10)
end

if test_cpu
  test_tied_inner_product_layer(backend_cpu)
end
if test_gpu
  test_tied_inner_product_layer(backend_gpu)
end
