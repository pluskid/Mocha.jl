function test_data_transformer(backend::Backend, T, eps)
  println("-- Testing DataTransformers on $(typeof(backend)){$T}...")

  input = rand(T, 4, 5, 6, 7)
  mean_data = rand(T, 4, 5, 6, 1)
  input_blob = make_blob(backend, input)
  mean_blob = make_blob(backend, mean_data)

  println("    > SubMean")
  trans = DataTransformers.SubMean(mean_blob = mean_blob)
  state = setup(backend, trans, input_blob)

  forward(backend, state, input_blob)
  transformed = similar(input)
  copy!(transformed, input_blob)

  input .-= mean_data
  @test all(abs.(transformed - input) .< eps)
  shutdown(backend, state)

  println("    > Scale")
  scale = rand()
  trans = DataTransformers.Scale(scale)
  state = setup(backend, trans, input_blob)
  forward(backend, state, input_blob)
  copy!(transformed, input_blob)

  input .*= scale
  @test all(abs.(transformed - input) .< eps)
  shutdown(backend, state)
end

function test_data_transformer(backend::Backend)
  test_data_transformer(backend, Float32, 1e-5)
  test_data_transformer(backend, Float64, 1e-10)
end

if test_cpu
  test_data_transformer(backend_cpu)
end
if test_gpu
  test_data_transformer(backend_gpu)
end


