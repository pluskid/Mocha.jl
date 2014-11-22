function test_data_transformer(sys::System, T, eps)
  println("-- Testing DataTransformers on $(typeof(sys.backend)){$T}...")

  input = rand(T, 4, 5, 6, 7)
  mean_data = rand(T, 4, 5, 6, 1)
  input_blob = make_blob(sys.backend, input)
  mean_blob = make_blob(sys.backend, mean_data)

  println("    > SubMean")
  trans = DataTransformers.SubMean(mean_blob = mean_blob)
  state = setup(sys, trans, input_blob)

  forward(sys, state, input_blob)
  transformed = similar(input)
  copy!(transformed, input_blob)

  input .-= mean_data
  @test all(abs(transformed - input) .< eps)
  shutdown(sys, state)

  println("    > Scale")
  scale = rand()
  trans = DataTransformers.Scale(scale)
  state = setup(sys, trans, input_blob)
  forward(sys, state, input_blob)
  copy!(transformed, input_blob)

  input .*= scale
  @test all(abs(transformed - input) .< eps)
  shutdown(sys, state)
end

function test_data_transformer(sys::System)
  test_data_transformer(sys, Float32, 1e-5)
  test_data_transformer(sys, Float64, 1e-10)
end

if test_cpu
  test_data_transformer(sys_cpu)
end
if test_cudnn
  test_data_transformer(sys_cudnn)
end


