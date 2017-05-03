function test_square_loss_layer(backend::Backend, T, eps)
  println("-- Testing SquareLossLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple(rand(6:11, tensor_dim)...)
  println("    > $dims")
  preds = rand(T, dims)
  labels = rand(T, dims)

  ############################################################
  # Setup
  ############################################################
  layer  = SquareLossLayer(; bottoms=[:predictions, :labels])
  pred_blob  = make_blob(backend, T, dims)
  label_blob = make_blob(backend, T, dims)
  diff_blob  = make_blob(backend, T, dims)
  copy!(pred_blob, preds)
  copy!(label_blob, labels)
  inputs = Blob[pred_blob, label_blob]
  diffs = Blob[diff_blob, NullBlob()]

  state  = setup(backend, layer, inputs, diffs)

  forward(backend, state, inputs)

  loss = 0.5*vecnorm(preds-labels)^2 / dims[end]
  @test -eps < loss-state.loss < eps

  backward(backend, state, inputs, diffs)
  grad = (preds - labels) / dims[end]
  diff = similar(grad)
  copy!(diff, diffs[1])

  @test all(-eps .< grad - diff .< eps)

  shutdown(backend, state)
end

function test_square_loss_layer(backend::Backend)
  test_square_loss_layer(backend, Float32, 1e-2)
  test_square_loss_layer(backend, Float64, 1e-8)
end

if test_cpu
  test_square_loss_layer(backend_cpu)
end
if test_gpu
  test_square_loss_layer(backend_gpu)
end
