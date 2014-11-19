function test_dropout_layer(sys::System, T, eps)
  println("-- Testing Dropout on $(typeof(sys.backend)){$T}...")

  ratio = convert(T, 0.6)
  input = rand(T, 3, 4, 5, 6)
  input_blob = make_blob(sys.backend, input)
  diff_blob  = make_blob(sys.backend, input)

  println("    > Setup")
  layer = DropoutLayer(tops=[:dropout], bottoms=[:input], ratio=ratio)
  state = setup(sys, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(sys, state, Blob[input_blob])
  got_output = zeros(T, size(input))
  copy!(got_output, state.blobs[1])
  rand_vals = zeros(T, size(input))
  copy!(rand_vals, state.rand_vals[1])
  expected_output = input .* (rand_vals .> ratio) / (1-ratio)
  @test all(abs(got_output - expected_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(input))
  copy!(state.blobs_diff[1], top_diff)
  backward(sys, state, Blob[input_blob], Blob[diff_blob])
  expected_grad = top_diff .* (rand_vals .> ratio) / (1-ratio)
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, diff_blob)
  @test all(abs(got_grad - expected_grad) .< eps)

  shutdown(sys, state)
end
function test_dropout_layer(sys::System)
  test_dropout_layer(sys, Float32, 1e-4)
  test_dropout_layer(sys, Float64, 1e-10)
end

if test_cpu
  test_dropout_layer(sys_cpu)
end
if test_cudnn
  test_dropout_layer(sys_cudnn)
end
