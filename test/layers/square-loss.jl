function test_square_loss_layer(sys::System)
  println("-- Testing SquareLossLayer...")

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
  layer  = SquareLossLayer(; bottoms=String["predictions", "labels"])
  if isa(sys.backend, CPUBackend)
    error("TODO")
  elseif isa(sys.backend, CuDNNBackend)
    pred_blob = Mocha.cudnn_make_tensor_blob(Float64, dims...)
    copy!(pred_blob, preds)
    label_blob = Mocha.cudnn_make_tensor_blob(Float64, dims...)
    copy!(label_blob, labels)
    inputs = Blob[pred_blob, label_blob]

    diffs = Blob[Mocha.cudnn_make_tensor_blob(Float64, dims...)]
  end
  state  = setup(sys, layer, inputs)

  forward(sys, state, inputs)

  loss = 0.5*vecnorm(preds-labels)^2 / dims[4]
  @test -eps < loss-state.loss < eps

  backward(sys, state, inputs, diffs)
  grad = (preds - labels) / dims[4]
  diff = similar(grad)
  copy!(diff, diffs[1])

  @test all(-eps .< grad - diff .< eps)
end

test_square_loss_layer(sys_cudnn)

