function test_inplace_layer(backend::Backend, T, eps)
  println("-- Testing InplaceLayer on $(typeof(backend)){$T}...")
  registry_reset(backend) # clear layer registry

  ratio = convert(T, 0.6)
  width, height, channels, batch_size = (3,4,5,6)
  input = rand(T, width, height, channels, batch_size)

  layer_data = MemoryDataLayer(tops=[:data], batch_size=batch_size,
      data = Array[input])
  layer_ip1 = InnerProductLayer(name="ip1", tops=[:ip1], bottoms=[:data], output_dim=8)
  layer_dropout1 = DropoutLayer(bottoms=[:ip1])
  layer_dropout2 = DropoutLayer(bottoms=[:ip1])
  layer_ip2 = InnerProductLayer(name="ip2", tops=[:ip2], bottoms=[:ip1], output_dim=1)

  # in random order but dropout1 is put before dropout2
  layers = [layer_ip2, layer_data, layer_ip1, layer_dropout1, layer_dropout2]

  println("    > Setup")
  net = Net("test-inplace", backend, layers)

  # make sure toplogical sort is working properly
  @test net.layers[1] == layer_data
  @test net.layers[2] == layer_ip1
  @test net.layers[3] == layer_dropout1
  @test net.layers[4] == layer_dropout2
  @test net.layers[5] == layer_ip2

  init(net)

  println("    > Forward")
  forward(net)

  ############################################################
  # manually do forward... =.=
  ############################################################
  #-- layer_ip1
  weight = zeros(T, size(net.states[2].W))
  copy!(weight, net.states[2].W)
  output = reshape(weight, size(weight,1),size(weight,2))' *
      reshape(input, width*height*channels, batch_size)

  #-- layer_dropout1
  rand_vals = zeros(T, size(output))
  copy!(rand_vals, net.states[3].rand_vals)
  output .*= (rand_vals .> layer_dropout1.ratio) / (1-layer_dropout1.ratio)

  #-- layer_dropout2
  copy!(rand_vals, net.states[4].rand_vals)
  output .*= (rand_vals .> layer_dropout2.ratio) / (1-layer_dropout2.ratio)
  output = convert(Array{T}, output)

  #-- middle-term check
  got_output = similar(output)
  copy!(got_output, net.states[2].blobs[1])
  @test all(abs.(output - got_output) .< eps)

  #-- layer_ip2
  weight = zeros(T, size(net.states[5].W))
  copy!(weight, net.states[5].W)
  output = reshape(weight, size(weight,1), size(weight,2))' * output
  got_output = similar(output)
  copy!(got_output, net.states[5].blobs[1])
  @test all(abs.(output - got_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(output))
  copy!(net.states[5].blobs_diff[1], top_diff)
  backward(net)
  ############################################################
  # manually do backward... =.=
  ############################################################
  #-- layer_ip2
  weight = zeros(T, size(net.states[5].W))
  copy!(weight, net.states[5].W)
  grad = reshape(weight, size(weight,1), size(weight,2)) * top_diff

  #-- layer_dropout2
  copy!(rand_vals, net.states[4].rand_vals)
  grad .*= (rand_vals .> layer_dropout2.ratio) / (1-layer_dropout2.ratio)

  #-- layer_dropout1
  copy!(rand_vals, net.states[3].rand_vals)
  grad .*= (rand_vals .> layer_dropout1.ratio) / (1-layer_dropout1.ratio)
  grad = convert(Array{T}, grad)

  got_grad = similar(grad)
  copy!(got_grad, net.states[2].blobs_diff[1])
  @test all(abs.(grad - got_grad) .< eps)

  destroy(net)
end
function test_inplace_layer(backend::Backend)
  test_inplace_layer(backend, Float64, 1e-10)
  test_inplace_layer(backend, Float32, 1e-4)
end

if test_cpu
  test_inplace_layer(backend_cpu)
end
if test_gpu
  test_inplace_layer(backend_gpu)
end

