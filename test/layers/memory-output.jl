function test_memory_output_layer(backend::Backend, T, eps)
  println("-- Testing Memory Output Layer on $(typeof(backend)){$T}...")

  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple(rand(1:8, tensor_dim)...)
  println("    > $dims")

  inputs = Array[rand(T, dims), rand(T, dims)]
  input_blobs = Blob[make_blob(backend, x) for x in inputs]

  layer = MemoryOutputLayer(bottoms=[:input1, :input2])
  state = setup(backend, layer, input_blobs, Array{Blob}(length(inputs)))

  # repeat 2 times
  forward(backend, state, input_blobs)
  forward(backend, state, input_blobs)

  @test length(state.outputs) == length(inputs)
  for i = 1:length(inputs)
    @test length(state.outputs[i]) == 2
    @test all(abs.(state.outputs[i][1] .- inputs[i]) .< eps)
    @test all(abs.(state.outputs[i][2] .- inputs[i]) .< eps)
  end

  reset_outputs(state)
  forward(backend, state, input_blobs)
  @test length(state.outputs) == length(inputs)
  for i = 1:length(inputs)
    @test length(state.outputs[i]) == 1
    @test all(abs.(state.outputs[i][1] .- inputs[i]) .< eps)
  end

  shutdown(backend, state)
end

function test_memory_output_layer(backend::Backend)
  test_memory_output_layer(backend, Float32, 1e-5)
  test_memory_output_layer(backend, Float64, 1e-10)
end

if test_cpu
  test_memory_output_layer(backend_cpu)
end
if test_gpu
  test_memory_output_layer(backend_gpu)
end

