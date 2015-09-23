function test_hinge_loss_layer(backend::Backend, T, eps)
  println("-- Testing HingeLossLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple((abs(rand(Int,tensor_dim)) % 6 + 6)...)
  println("    > $dims")
  preds = rand(T, dims)*4-2
  labels = round(rand(T, dims))*2-1
  errs_mask = preds.*labels .< one(T)

  ############################################################
  # Setup
  ############################################################
  layer  = HingeLossLayer(; bottoms=[:predictions, :labels])
  pred_blob  = make_blob(backend, T, dims)
  label_blob = make_blob(backend, T, dims)
  diff_blob  = make_blob(backend, T, dims)
#   diff_blob2 = make_blob(backend, T, dims)
  copy!(pred_blob, preds)
  copy!(label_blob, labels)
  inputs = Blob[pred_blob, label_blob]
  diffs  = Blob[diff_blob, NullBlob()]
#   diffs2 = Blob[diff_blob, diff_blob2]

  state  = setup(backend, layer, inputs, diffs)

  forward(backend, state, inputs)

  loss = sum(max(one(T) .- preds.*labels, zero(T))) / dims[end]
  @test -eps < loss-state.loss < eps

  backward(backend, state, inputs, diffs)
  grad  = -labels .* errs_mask / dims[end]
#   grad2 = -preds  .* errs_mask / dims[end]
  diff  = similar(grad)
  copy!(diff, diffs[1])

  @test all(-eps .< grad - diff .< eps)

  shutdown(backend, state)
end

function test_hinge_loss_layer(backend::Backend)
  test_hinge_loss_layer(backend, Float32, 1e-2)
  test_hinge_loss_layer(backend, Float64, 1e-8)
end

if test_cpu
  test_hinge_loss_layer(backend_cpu)
end
if test_gpu
  test_hinge_loss_layer(backend_gpu)
end
