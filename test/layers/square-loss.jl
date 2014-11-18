function test_square_loss_layer(sys::System)
  println("-- Testing SquareLossLayer on $(typeof(sys.backend))...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  dims = (20,3,3,40)
  preds = rand(dims)
  labels = rand(dims)
  eps          = 1e-10

  ############################################################
  # Setup
  ############################################################
  layer  = SquareLossLayer(; bottoms=[:predictions, :labels])
  if isa(sys.backend, CPUBackend)
    pred_blob  = CPUBlob(Float64, dims...)
    label_blob = CPUBlob(Float64, dims...)
    diff_blob  = CPUBlob(Float64, dims...)
  elseif isa(sys.backend, CuDNNBackend)
    pred_blob  = Mocha.cudnn_make_tensor_blob(Float64, dims...)
    label_blob = Mocha.cudnn_make_tensor_blob(Float64, dims...)
    diff_blob  = Mocha.cudnn_make_tensor_blob(Float64, dims...)
  end
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

if test_cpu
  test_square_loss_layer(sys_cpu)
end
if test_cudnn
  test_square_loss_layer(sys_cudnn)
end
