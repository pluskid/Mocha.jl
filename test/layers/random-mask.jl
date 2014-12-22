function test_random_mask_layer(backend, T, eps)
  println("-- Testing RandomMask on $(typeof(backend)){$T}")

  n_inputs = 3
  tensor_dims = [abs(rand(Int)) % 6 + 1 for i = 1:n_inputs]
  dims_all = [tuple((abs(rand(Int, tensor_dims[i])) % 8 + 1) ...) for i = 1:n_inputs]
  println("    > $n_inputs input blobs with tensor dims $tensor_dims")

  ratio = min(abs(rand()) + 0.1, 0.9)
  inputs = Array[rand(T, dims_all[i]) for i = 1:n_inputs]
  input_blobs = Blob[make_blob(backend, inputs[i]) for i = 1:n_inputs]
  diff_blobs = Blob[make_blob(backend, inputs[i]) for i = 1:n_inputs]

  println("    > Setup")
  layer = RandomMaskLayer(bottoms=[symbol("inputs-$i") for i = 1:n_inputs], ratio=ratio)
  state = setup(backend, layer, input_blobs, diff_blobs)

  println("    > Forward")
  forward(backend, state, input_blobs)
  for i = 1:n_inputs
    rand_vals = to_array(state.dropouts[i].rand_vals)
    expected_output = inputs[i] .* (rand_vals .> ratio)
    got_output = to_array(input_blobs[i])
    @test all(abs(got_output - expected_output) .< eps)
  end

  println("    > Backward")
  top_diffs = [rand(T, dims_all[i]) for i = 1:n_inputs]
  for i = 1:n_inputs
    copy!(diff_blobs[i], top_diffs[i])
  end
  backward(backend, state, input_blobs, diff_blobs)
  for i = 1:n_inputs
    rand_vals = to_array(state.dropouts[i].rand_vals)
    got_grad = to_array(diff_blobs[i])
    expected_grad = top_diffs[i] .* (rand_vals .> ratio)
    @test all(abs(got_grad - expected_grad) .< eps)
  end

  shutdown(backend, state)
end
function test_random_mask_layer(backend::Backend)
  test_random_mask_layer(backend, Float64, 1e-10)
  test_random_mask_layer(backend, Float32, 1e-4)
end

if test_cpu
  test_random_mask_layer(backend_cpu)
end
if test_gpu
  test_random_mask_layer(backend_gpu)
end
