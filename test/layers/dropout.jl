function test_dropout_layer(backend::Backend, T, eps)
  println("-- Testing Dropout on $(typeof(backend)){$T}...")

  tensor_dim = abs(rand(Int)) % 6 + 1
  dims = tuple(rand(1:8, tensor_dim)...)
  println("    > $dims")

  ratio = convert(T, 0.6)
  input = rand(T, dims)
  input_blob = make_blob(backend, input)
  diff_blob  = make_blob(backend, input)

  println("    > Setup")
  layer = DropoutLayer(bottoms=[:input], ratio=ratio)
  state = setup(backend, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(input))
  copy!(got_output, input_blob)
  rand_vals = zeros(T, size(input))
  copy!(rand_vals, state.rand_vals)
  expected_output = input .* (rand_vals .> ratio) / (1-ratio)
  @test all(abs.(got_output - expected_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(input))
  copy!(diff_blob, top_diff)
  backward(backend, state, Blob[input_blob], Blob[diff_blob])
  expected_grad = top_diff .* (rand_vals .> ratio) / (1-ratio)
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, diff_blob)
  @test all(abs.(got_grad - expected_grad) .< eps)

  shutdown(backend, state)
end
function test_dropout_layer(backend::Backend)
  test_dropout_layer(backend, Float64, 1e-10)
  test_dropout_layer(backend, Float32, 1e-4)
end

if test_cpu
  test_dropout_layer(backend_cpu)
end
if test_gpu
  test_dropout_layer(backend_gpu)
end
