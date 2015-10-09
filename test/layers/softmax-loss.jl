function test_softmax_loss_layer(backend::Backend, tensor_dim, use_weights::Bool, T, eps)
  println("-- Testing SoftmaxLossLayer on $(typeof(backend)){$T} $(use_weights ? "(with weights)" : "")...")

  if use_weights
    op_dim = tensor_dim-1
  else
    op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  end
  dims = abs(rand(Int,tensor_dim)) % 6 + 6
  dims_label = copy(dims); dims_label[op_dim] = 1
  dims = tuple(dims...)
  dims_label = tuple(dims_label...)
  println("    > $dims (operate on dimension $op_dim)")

  input = rand(T, dims) + 0.01

  input_blob = make_blob(backend, input)
  diff_blob = make_blob(backend, T, size(input))

  if use_weights
    weights = rand(T, dims[1:end-1]) + 0.1
    weights = weights .* (dims[op_dim] ./ sum(weights,op_dim))
  else
    weights = []
  end

  label = abs(rand(Int, dims_label)) % dims[op_dim]
  label = convert(Array{T}, label)
  label_blob = make_blob(backend, label)

  inputs = Blob[input_blob, label_blob]

  loss_weight = abs(rand(T))

  layer = SoftmaxLossLayer(bottoms=[:pred, :labels], weights=weights, normalize=:local, dim=op_dim, weight=loss_weight)
  state = setup(backend, layer, inputs, Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, inputs)

  new_shape = [size(weights)..., 1]
  rep_shape = ones(Int, tensor_dim); rep_shape[end] = dims[end]
  weights = repeat(reshape(weights, new_shape...), inner=rep_shape)

  expected_loss = 0
  expected_grad = zeros(T, size(input))

  dim_pre, dim_prob, dim_post = split_dims(input, op_dim)
  canonical_input = reshape(input, dim_pre, dim_prob, dim_post)
  canonical_grad = reshape(expected_grad, dim_pre, dim_prob, dim_post)
  if !isempty(weights)
    weights = reshape(weights, (dim_pre, dim_prob, dim_post))
  end
  label = reshape(label, dim_pre, 1, dim_post)
  for i = 1:dim_pre
    for j = 1:dim_post
      pred = exp(canonical_input[i,:,j])
      pred /= sum(pred)
      if isempty(weights)
        canonical_grad[i,:,j] = pred
        canonical_grad[i,round(Int, label[i,1,j])+1,j] -= 1
        expected_loss += -log(pred[round(Int, label[i,1,j])+1])
      else
        y = round(Int, label[i,1,j])+1
        canonical_grad[i,:,j] = pred .* weights[i,y,j]
        canonical_grad[i,y,j] -= weights[i,y,j]
        expected_loss += -log(pred[y]) * weights[i,y,j]
      end
    end
  end
  scale = dims[op_dim] / prod(dims)
  expected_loss *= scale * loss_weight
  expected_grad *= scale * loss_weight
  expected_grad = reshape(expected_grad, size(input))

  @test -eps < state.loss - expected_loss < eps

  println("    > Backward")
  backward(backend, state, inputs, Blob[diff_blob])
  grad = zeros(T, size(input))
  copy!(grad, diff_blob)

  @test all(-eps .< grad - expected_grad .< eps)

  shutdown(backend, state)
end
function test_softmax_loss_layer(backend::Backend, use_weights::Bool, T, eps)
  for i in [2,4,5]
    test_softmax_loss_layer(backend, i, use_weights, T, eps)
  end
end
function test_softmax_loss_layer(backend::Backend, T, eps)
  test_softmax_loss_layer(backend, false, T, eps)
  test_softmax_loss_layer(backend, true, T, eps)
end
function test_softmax_loss_layer(backend::Backend)
  test_softmax_loss_layer(backend, Float64, 1e-5)
  test_softmax_loss_layer(backend, Float32, 1e-3)
end

if test_cpu
  test_softmax_loss_layer(backend_cpu)
end
if test_gpu
  test_softmax_loss_layer(backend_gpu)
end
