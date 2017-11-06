function test_accuracy_layer(backend::Backend, tensor_dim, T)
  println("-- Testing AccuracyLayer on $(typeof(backend)){$T}...")

  dims = rand(6:11, tensor_dim)
  op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  dims_label = copy(dims); dims_label[op_dim] = 1
  dims = tuple(dims...)
  dims_label = tuple(dims_label...)
  println("    > $dims (operate on dimension $op_dim)")

  eps = 1e-5
  input = rand(T, dims)
  input_blob = make_blob(backend, input)

  label = abs.(rand(Int, dims_label)) .% dims[op_dim]
  label = convert(Array{T}, label)
  label_blob = make_blob(backend, label)

  inputs = Blob[input_blob, label_blob]

  layer = AccuracyLayer(bottoms=[:pred, :labels], dim=op_dim)
  state = setup(backend, layer, inputs, Blob[])

  println("    > Forward")
  forward(backend, state, inputs)

  @test state.n_accum == prod(dims_label)

  dim_pre, dim_pred, dim_post = split_dims(input, op_dim)

  canonical_input = reshape(input, (dim_pre, dim_pred, dim_post))
  canonical_label = reshape(label, (dim_pre, 1, dim_post))
  expected_acc = 0.0
  for i = 1:dim_pre
    for j = 1:dim_post
      pred = canonical_input[i,:,j]
      if indmax(pred) == round(Int, canonical_label[i,1,j])+1
        expected_acc += 1
      end
    end
  end
  expected_acc /= prod(dims_label)
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again")
  forward(backend, state, inputs)
  @test state.n_accum == 2*prod(dims_label)
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again and Again")
  reset_statistics(state)
  forward(backend, state, inputs)
  @test state.n_accum == prod(dims_label)
  @test abs(state.accuracy - expected_acc) < eps

  shutdown(backend, state)
end
function test_accuracy_layer(backend::Backend, T)
  for i in [2,4,5]
    test_accuracy_layer(backend, i, T)
  end
end
function test_accuracy_layer(backend::Backend)
  test_accuracy_layer(backend, Float32)
  test_accuracy_layer(backend, Float64)
end

if test_cpu
  test_accuracy_layer(backend_cpu)
end
if test_gpu
  test_accuracy_layer(backend_gpu)
end
