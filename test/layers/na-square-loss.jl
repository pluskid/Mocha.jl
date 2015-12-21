function test_na_square_loss_layer(backend::Backend,NAvalue, T, eps)
  println("-- Testing NaSquareLossLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  NAvalue = convert(T, NAvalue)
  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple((abs(rand(Int,tensor_dim)) % 6 + 6)...)
  println("    > $dims")
  preds = rand(T, dims)
  labels = rand(T, dims)

  napos = unique( sort( rand( eachindex(labels), cld( length(labels) , 2 ) ) ) )
  labels[napos] = NAvalue

  ############################################################
  # Setup
  ############################################################
  layer  = NaSquareLossLayer(;NAvalue = NAvalue, bottoms=[:predictions, :labels])
  pred_blob  = make_blob(backend, T, dims)
  label_blob = make_blob(backend, T, dims)
  diff_blob  = make_blob(backend, T, dims)
  copy!(pred_blob, preds)
  copy!(label_blob, labels)
  inputs = Blob[pred_blob, label_blob]
  diffs = Blob[diff_blob, NullBlob()]

  state  = setup(backend, layer, inputs, diffs)

  forward(backend, state, inputs)

  dd = preds .- labels
  Mocha.rmna!(labels, dd, NAvalue)
  loss = 0.5*vecnorm(dd)^2 / dims[end]
  @test -eps < loss-state.loss < eps

  backward(backend, state, inputs, diffs)
  grad = dd / dims[end]
  diff = similar(grad)
  copy!(diff, diffs[1])

  @test all(-eps .< grad - diff .< eps)

  shutdown(backend, state)
end

function test_na_square_loss_layer(backend::Backend)
  test_na_square_loss_layer(backend, NaN32,     Float32, 1e-2)
  test_na_square_loss_layer(backend, NaN64,     Float64, 1e-8)
  test_na_square_loss_layer(backend, -9999.0f0, Float32, 1e-2)
  test_na_square_loss_layer(backend, -9999.0,   Float64, 1e-8)
end

if test_cpu
  test_na_square_loss_layer(backend_cpu)
end
if test_gpu
  test_na_square_loss_layer(backend_gpu)
end
