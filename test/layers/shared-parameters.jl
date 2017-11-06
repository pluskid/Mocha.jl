function test_shared_parameters_layers(backend::Backend, layer_type, T, eps)
  println("-- Testing $layer_type layer with shared param on $(typeof(backend)){$T}...")
  registry_reset(backend)

  w,h,c,n = 2,3,4,5
  input = rand(T, w,h,c,n)

  layer_data = MemoryDataLayer(tops=[:data], batch_size=n, data=Array[input])
  layer_split = SplitLayer(tops=[:data1,:data2], bottoms=[:data])

  key="key"
  if layer_type == "convolution"
    l1 = ConvolutionLayer(name="conv1", param_key=key, neuron=Neurons.Sigmoid(), tops=[:out1], bottoms=[:data1])
    l2 = ConvolutionLayer(name="conv2", param_key=key, neuron=Neurons.Sigmoid(), tops=[:out2], bottoms=[:data2])
  elseif layer_type == "inner-product"
    out_dim = 5
    l1 = InnerProductLayer(name="ip1", output_dim=out_dim, neuron=Neurons.ReLU(), param_key=key, tops=[:out1], bottoms=[:data1])
    l2 = InnerProductLayer(name="ip2", output_dim=out_dim, neuron=Neurons.ReLU(), param_key=key, tops=[:out2], bottoms=[:data2])
  else
    error("Unknown layer_type $layer_type")
  end

  layer_sub = ElementWiseLayer(operation=ElementWiseFunctors.Subtract(),bottoms=[:out1,:out2],tops=[:diff])

  net = Net("test-shared-params", backend, [layer_data, layer_split, l1, l2, layer_sub])
  init(net)
  forward(net)

  @test net.layers[end] == layer_sub
  output = zeros(T, size(net.states[end].blobs[1]))
  copy!(output, net.states[end].blobs[1])
  @test all(abs.(output) .< eps)

  destroy(net)
end

function test_shared_parameters_layers(backend::Backend, T, eps)
  test_shared_parameters_layers(backend, "convolution", T, eps)
  test_shared_parameters_layers(backend, "inner-product", T, eps)
end
function test_shared_parameters_layers(backend::Backend)
  test_shared_parameters_layers(backend, Float64, 1e-9)
  test_shared_parameters_layers(backend, Float32, 1e-4)
end

if test_cpu
  test_shared_parameters_layers(backend_cpu)
end
if test_gpu
  test_shared_parameters_layers(backend_gpu)
end
