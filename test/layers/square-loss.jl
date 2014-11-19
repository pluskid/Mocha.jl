function test_square_loss_layer(sys::System, T, eps)
  println("-- Testing SquareLossLayer on $(typeof(sys.backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  dims = (20,3,3,40)
  preds = rand(T, dims)
  labels = rand(T, dims)

  ############################################################
  # Setup
  ############################################################
  layer  = SquareLossLayer(; bottoms=[:predictions, :labels])
  pred_blob  = make_blob(sys.backend, T, dims...)
  label_blob = make_blob(sys.backend, T, dims...)
  diff_blob  = make_blob(sys.backend, T, dims...)
  copy!(pred_blob, preds)
  copy!(label_blob, labels)
  inputs = Blob[pred_blob, label_blob]
  diffs = Blob[diff_blob]

  state  = setup(sys, layer, inputs, diffs)

  forward(sys, state, inputs)

  loss = 0.5*vecnorm(preds-labels)^2 / dims[4]
  @test -eps < loss-state.loss < eps

  backward(sys, state, inputs, diffs)
  grad = (preds - labels) / dims[4]
  diff = similar(grad)
  copy!(diff, diffs[1])

  @test all(-eps .< grad - diff .< eps)

  shutdown(sys, state)
end

function test_square_loss_layer(sys::System)
  test_square_loss_layer(sys, Float32, 1e-3)
  test_square_loss_layer(sys, Float64, 1e-10)
end

if test_cpu
  test_square_loss_layer(sys_cpu)
end
if test_cudnn
  test_square_loss_layer(sys_cudnn)
end
