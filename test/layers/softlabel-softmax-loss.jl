function test_softlabel_softmax_loss_layer(backend::Backend, tensor_dim, T, eps)
  println("-- Testing SoftlabelSoftmaxLossLayer on $(typeof(backend)){$T}...")

  dims = tuple(rand(6:11, tensor_dim)...)
  op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  println("    > $dims (operate on dimension $op_dim)")

  input = rand(T, dims) + convert(T, 0.01)
  input_blob = make_blob(backend, input)
  diff_blob = make_blob(backend, T, size(input))

  labels = abs.(rand(T, dims)) + convert(T, 0.01)
  labels = labels ./ sum(labels, op_dim)
  label_blob = make_blob(backend, labels)

  inputs = Blob[input_blob, label_blob]

  loss_weight = abs(rand(T))
  layer = SoftlabelSoftmaxLossLayer(bottoms=[:pred, :labels], dim=op_dim, weight=loss_weight)
  state = setup(backend, layer, inputs, Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, inputs)

  expected_loss = 0
  expected_grad = zeros(T, size(input))

  dim_pre, dim_prob, dim_post = split_dims(input, op_dim)
  canonical_input = reshape(input, dim_pre, dim_prob, dim_post)
  canonical_label = reshape(labels, dim_pre, dim_prob, dim_post)
  canonical_grad = reshape(expected_grad, dim_pre, dim_prob, dim_post)

  for i = 1:dim_pre
    for j = 1:dim_post
      pred = exp.(canonical_input[i,:,j])
      pred /= sum(pred)

      expected_loss += sum(-log.(pred) .* canonical_label[i,:,j])
      canonical_grad[i,:,j] = repmat(vec(pred), 1, dim_prob) * vec(canonical_label[i,:,j])
      canonical_grad[i,:,j] -= canonical_label[i,:,j]
    end
  end
  scale = dims[op_dim] / prod(dims)
  expected_loss *= scale * loss_weight
  expected_grad *= scale * loss_weight
  expected_grad = reshape(expected_grad, size(input))

  @test -eps < state.loss - expected_loss < eps

  println("    > Backward")
  backward(backend, state, inputs, Blob[diff_blob])
  grad = to_array(diff_blob)

  @test all(-eps .< grad - expected_grad .< eps)

  shutdown(backend, state)
end

function test_softlabel_softmax_loss_layer(backend::Backend, T, eps)
  for i in [2,4,5]
    test_softlabel_softmax_loss_layer(backend, i, T, eps)
  end
end
function test_softlabel_softmax_loss_layer(backend::Backend)
  test_softlabel_softmax_loss_layer(backend, Float64, 1e-5)
  test_softlabel_softmax_loss_layer(backend, Float32, 1e-3)
end
if test_cpu
  test_softlabel_softmax_loss_layer(backend_cpu)
end
if test_gpu
  test_softlabel_softmax_loss_layer(backend_gpu)
end
