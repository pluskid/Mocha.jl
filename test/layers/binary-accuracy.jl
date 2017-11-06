function test_binary_accuracy_layer(backend::Backend, tensor_dim, T, threshold, eps)
  println("-- Testing BinaryAccuracyLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  dims = tuple(rand(6:11, tensor_dim)...)
  println("    > $dims")
  if 0 == threshold
    preds = rand(T, dims)*4-2
    labels = round.(rand(T, dims))*2-1
  elseif 0.5 == threshold
    preds = round.(rand(T, dims))
    labels = rand(T, dims)
  else
    error("Threshold must be 0 or 0.5; was $threshold")
  end
  errs_mask = (preds.>threshold) .≠ (labels.>threshold)
  n = prod(dims)
  n_wrong = countnz(errs_mask)
  n_right = n - n_wrong

  ############################################################
  # Setup
  ############################################################
  layer = BinaryAccuracyLayer(; bottoms=[:predictions, :labels], threshold=threshold)
  pred_blob  = make_blob(backend, T, dims)
  label_blob = make_blob(backend, T, dims)
  copy!(pred_blob, preds)
  copy!(label_blob, labels)
  inputs = Blob[pred_blob, label_blob]

  state  = setup(backend, layer, inputs, Blob[])

  ###########################################################
  # Forward
  ###########################################################
  println("    > Forward")
  forward(backend, state, inputs)

  @test state.n_accum == n

  expected_acc = convert(T, n_right / n)
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again")
  forward(backend, state, Blob[label_blob, label_blob])
  @test state.n_accum == 2n
  @test abs(state.accuracy - (expected_acc+1)/2) < eps

  println("    > Forward Again and Again")
  reset_statistics(state)
  forward(backend, state, inputs)
  @test state.n_accum == n
  @test abs(state.accuracy - expected_acc) < eps

  shutdown(backend, state)
end
function test_binary_accuracy_layer(backend::Backend, T, eps)
  for i in [2,4,5]
    for threshold in [0, 0.5]
      test_binary_accuracy_layer(backend, i, T, threshold, eps)
    end
  end
end
function test_binary_accuracy_layer(backend::Backend)
  test_binary_accuracy_layer(backend, Float32, 1e-4)
  test_binary_accuracy_layer(backend, Float64, 1e-8)
end

if test_cpu
  test_binary_accuracy_layer(backend_cpu)
end
if test_gpu
  test_binary_accuracy_layer(backend_gpu)
end
