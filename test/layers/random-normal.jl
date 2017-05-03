function test_random_normal_layer(backend::Backend, T, eps)
  println("-- Testing RandomNormal Layer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################

  batch_sizes = [100,200]
  N = length(batch_sizes)
  tensor_dim = rand(1:4)
  output_dims = rand(1:3, tensor_dim)
  println("    > Random output ", tensor_dim, output_dims)


  ############################################################
  # Setup
  ############################################################

  # batch size is determined by
  layer = RandomNormalLayer(tops = [:data1, :data2], output_dims=output_dims,
                            eltype=T, batch_sizes=batch_sizes)
  state = setup(backend, layer, Blob[], Blob[])

  layer_data = [Array{T}(tuple(output_dims..., batch_sizes[i]))
                for i in 1:N]

  forward(backend, state, Blob[])
  for i in 1:N
      copy!(layer_data[i], state.blobs[i])
      @test (abs(mean(layer_data[i])) < 4e-1)
      @test all(-1000 .< layer_data[i] .< 1000)      
  end

    # we should have sample from zero mean, unit stddev in state.blobs[1]
    # the size should be as expected


    # output should be different on subsequent calls
  layer_data2 = [Array{T}(tuple(output_dims..., batch_sizes[i]))
                for i in 1:N]
                    
  forward(backend, state, Blob[])
  for i in 1:N
      copy!(layer_data2[i], state.blobs[i])
      @test (abs(mean(layer_data2[i])) < 4e-1)
      @test all(-1000 .< layer_data2[i] .< 1000)
      @test 0 < norm(vec(layer_data[i] - layer_data2[i])) < 5sqrt(prod(output_dims)*batch_sizes[i])
  end


  shutdown(backend, state)
end
function test_random_normal_layer(backend::Backend)
  test_random_normal_layer(backend, Float32, 1e-5)
  test_random_normal_layer(backend, Float64, 1e-10)
end


if test_gpu
  test_random_normal_layer(backend_gpu)
end
 
if test_cpu
  test_random_normal_layer(backend_cpu)
end
