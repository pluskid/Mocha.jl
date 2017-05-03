function test_binary_crossentropy_loss_layer(backend::Backend, tensor_dim, T, eps)
  println("-- Testing BinaryCrossEntropyLossLayer on $(typeof(backend)){$T}...")

  dims = rand(2:7, tensor_dim)


  println("    > $dims")

  #dims_label = copy(dims); dims_label[op_dim] = 1; dims_label = tuple(dims_label...)
  dims = tuple(dims...)

  prob = rand(T, dims)

  label = rand(Int, dims) .> 0.5
  label = convert(Array{T}, label)

  prob_blob = make_blob(backend, prob)
  label_blob = make_blob(backend, label)
  inputs = Blob[prob_blob, label_blob]
  weight = T(0.25)
  layer = BinaryCrossEntropyLossLayer(bottoms=[:pred, :labels], weight=weight)
  state = setup(backend, layer, inputs, Blob[])

  forward(backend, state, inputs)

  expected_loss = convert(T, 0)

  for i = 1:prod(dims)
      expected_loss += -log(vec(prob)[i])*vec(label)[i]
      expected_loss += -log(1 - vec(prob)[i])*vec(1 - label)[i]
  end

  expected_loss /= dims[end]
  expected_loss *= weight

  @test -eps < 1 - state.loss/expected_loss < eps

  diff_blob2  = make_blob(backend, prob)
  diff_blob1  = make_blob(backend, prob)
  diffs = Blob[diff_blob1, diff_blob2]
  backward(backend, state, inputs, diffs)
  grad_pred = -weight * (label./prob - (1-label)./(1-prob) ) / dims[end]
  diff = similar(grad_pred)

  copy!(diff, diffs[1])

  @test all(-eps .< 1 - grad_pred./diff .< eps)

  grad_label = -weight * log.(prob./(1.-prob)) / dims[end]
  diff = similar(grad_pred)
  copy!(diff, diffs[2])

  @test all(-eps .< grad_label - diff .< eps)

  shutdown(backend, state)
end

function test_binary_crossentropy_loss_layer(backend::Backend, T, eps)
  for i in [2,4,5]
      test_binary_crossentropy_loss_layer(backend, i, T, eps)
  end
end

function test_binary_crossentropy_loss_layer(backend::Backend)
  test_binary_crossentropy_loss_layer(backend, Float32, 1e-5)
  test_binary_crossentropy_loss_layer(backend, Float64, 1e-6)
end

if test_cpu
  test_binary_crossentropy_loss_layer(backend_cpu)
end
if test_gpu
  test_binary_crossentropy_loss_layer(backend_gpu)
end
