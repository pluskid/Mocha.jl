function test_concat_layer(backend::Backend, dim, T, eps)
  println("-- Testing ConcatLayer(dim=$dim) on $(typeof(backend)){$T}...")

  tensor_dim = max(abs(rand(Int)) % 6 + 2, dim)
  println("    > $tensor_dim-dimensional tensor")

  n_input = 3
  dims_proto = rand(1:6, tensor_dim)
  dims = [copy(dims_proto) for i = 1:n_input]
  for i = 1:n_input
    dims[i][dim] = abs(rand(Int)) % 5 + 1
  end
  dims = map(x -> tuple(x...), dims)

  inputs = [rand(T, dims[i]) for i = 1:n_input]
  input_blobs = Blob[make_blob(backend, x) for x in inputs]
  grad_blobs = Blob[make_blob(backend, x) for x in inputs]

  layer = ConcatLayer(dim=dim, bottoms=Array{Symbol}(n_input), tops=[:concat])
  state = setup(backend, layer, input_blobs, grad_blobs)

  println("    > Forward")
  forward(backend, state, input_blobs)

  expected_output = cat(dim, inputs...)
  got_output = to_array(state.blobs[1])
  @test all(abs.(expected_output-got_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(expected_output))
  copy!(state.blobs_diff[1], top_diff)
  backward(backend, state, input_blobs, grad_blobs)
  for i = 1:n_input
    copy!(inputs[i], grad_blobs[i])
  end
  got_grad_all = cat(dim, inputs...)
  @test all(abs.(top_diff - got_grad_all) .< eps)
end

function test_concat_layer(backend::Backend, T, eps)
  for dim = 1:5
    test_concat_layer(backend, dim, T, eps)
  end
end

function test_concat_layer(backend::Backend)
  test_concat_layer(backend, Float64, 1e-10)
  test_concat_layer(backend, Float32, 1e-5)
end

if test_cpu
  test_concat_layer(backend_cpu)
end
if test_gpu
  test_concat_layer(backend_gpu)
end
