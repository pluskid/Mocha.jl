function test_multinomial_logistic_loss_layer(backend::Backend, tensor_dim, class_weights, T, eps)
  println("-- Testing MultinomialLogisticLossLayer{$(class_weights[1]),$(class_weights[2])} on $(typeof(backend)){$T}...")

  dims = rand(6:11, tensor_dim)
  if class_weights[1] != :no
    op_dim = tensor_dim-1
  else
    op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  end
  println("    > $dims (operate on dimension $op_dim)")

  dims_label = copy(dims); dims_label[op_dim] = 1; dims_label = tuple(dims_label...)
  dims = tuple(dims...)
  channels = dims[op_dim]

  prob = abs.(rand(T, dims)) .+ 0.01

  label = abs.(rand(Int, dims_label)) .% channels
  label = convert(Array{T}, label)

  prob_blob = make_blob(backend, prob)
  label_blob = make_blob(backend, label)
  inputs = Blob[prob_blob, label_blob]

  if class_weights[1] == :equal
    weights = ones(T, channels)
  elseif class_weights[1] == :local
    weights = rand(T, channels)
  elseif class_weights[1] == :global
    weights = round.(1000*rand(T, dims[1:end-1]))/1000
  else
    @assert class_weights[1] == :no
    weights = []
  end

  layer = MultinomialLogisticLossLayer(bottoms=[:pred, :labels],
      weights=weights, normalize=class_weights[2], dim=op_dim)
  state = setup(backend, layer, inputs, Blob[])

  forward(backend, state, inputs)

  if class_weights[1] == :local || class_weights[1] == :equal
    new_shape = ones(Int, tensor_dim-1); new_shape[op_dim] = dims[op_dim]
    rep_shape = [dims[1:end-1]...]; rep_shape[op_dim] = 1
    weights = repeat(reshape(weights, new_shape...), inner=rep_shape)
  end
  if class_weights[2] == :local
    weights = weights .* (channels ./ sum(weights,op_dim))
  elseif class_weights[2] == :global
    weights = weights * (prod(dims[1:end-1]) / sum(weights))
  else
    @assert class_weights[2] == :no
  end

  new_shape = [size(weights)..., 1]
  rep_shape = ones(Int, tensor_dim); rep_shape[end] = dims[end]
  weights = repeat(reshape(weights, new_shape...), inner=rep_shape)

  expected_loss = convert(T, 0)
  dim_pre, dim_prob, dim_post = split_dims(prob, op_dim)
  prob = reshape(prob, (dim_pre, dim_prob, dim_post))
  if !isempty(weights)
    weights = reshape(weights, (dim_pre, dim_prob, dim_post))
  end
  label = reshape(label, (dim_pre, 1, dim_post))
  for i = 1:dim_pre
    for j = 1:dim_post
      if isempty(weights)
        expected_loss += -log(prob[i, round(Int, label[i,1,j])+1, j])
      else
        idx = round(Int, label[i,1,j])+1
        expected_loss += -log(prob[i,idx,j]) * weights[i,idx,j]
      end
    end
  end
  expected_loss /= prod(dims) / dims[op_dim]

  @test -eps < state.loss - expected_loss < eps

  shutdown(backend, state)
end
function test_multinomial_logistic_loss_layer(backend::Backend, class_weights, T, eps)
  for i in [2,4,5]
    test_multinomial_logistic_loss_layer(backend, i, class_weights, T, eps)
  end
end
function test_multinomial_logistic_loss_layer(backend::Backend, T, eps)
  for class_weights in ((:equal,:local),(:local,:local),(:global,:global),(:global,:local),(:no,:no))
    test_multinomial_logistic_loss_layer(backend, class_weights, T, eps)
  end
end
function test_multinomial_logistic_loss_layer(backend::Backend)
  test_multinomial_logistic_loss_layer(backend, Float64, 1e-5)
  if !test_gpu || backend != backend_gpu
    # only test float32 on CPU, it seems the discrepancy from float32 is too big here
    test_multinomial_logistic_loss_layer(backend, Float32, 1e-3)
  end
end

if test_cpu
  test_multinomial_logistic_loss_layer(backend_cpu)
end
if test_gpu
  test_multinomial_logistic_loss_layer(backend_gpu)
end

