function test_inner_product_layer(backend::Backend, n_input, T, eps)
  println("-- Testing InnerProductLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  # inner-product layer should be able to take blobs with
  # different num, but the input feature dimensions should
  # be the same
  batch_size   = [abs(rand(Int)) % 50 + 1 for i = 1:n_input]
  orig_dim_all = (10, 20, 3)
  orig_dim     = prod(orig_dim_all)
  target_dim   = 30

  X = [rand(T, orig_dim_all..., n) for n in batch_size]
  W = rand(T, orig_dim, target_dim)
  b = rand(T, target_dim)

  ############################################################
  # Setup
  ############################################################
  layer = InnerProductLayer(name="ip", output_dim=target_dim,
      tops=Array{Symbol}(n_input), bottoms=Array{Symbol}(n_input))
  inputs = Blob[make_blob(backend, x) for x in X]
  diffs = Blob[make_blob(backend, x) for x in X]

  println("    > Setup")
  state  = setup(backend, layer, inputs, diffs)

  @test length(state.W) == length(W)
  @test length(state.b) == length(b)
  copy!(state.W, W)
  copy!(state.b, b)

  println("    > Forward")
  forward(backend, state, inputs)

  X2 = [reshape(X[i], orig_dim, batch_size[i]) for i = 1:n_input]
  res = [W' * x2 .+ b for x2 in X2]

  res_layer = [similar(x) for x in res]
  for i = 1:n_input
    copy!(res_layer[i], state.blobs[i])
    @test all(-eps .< res_layer[i] - res[i] .< eps)
  end

  println("    > Backward")
  top_diff = [rand(T, size(state.blobs_diff[i])) for i = 1:n_input]
  for i = 1:n_input
    copy!(state.blobs_diff[i], top_diff[i])
  end
  backward(backend, state, inputs, diffs)

  top_diff = [reshape(top_diff[i], target_dim, batch_size[i]) for i = 1:n_input]
  bias_grad = similar(b)
  copy!(bias_grad, state.∇b)
  bias_grad_expected = sum([sum(top_diff[i],2) for i = 1:n_input])
  @test all(-eps .< vec(bias_grad) - vec(bias_grad_expected) .< eps)

  X_mat = [reshape(X[i], orig_dim, batch_size[i]) for i = 1:n_input]
  weight_grad = similar(W)
  copy!(weight_grad, state.∇W)
  weight_grad_expected = sum([X_mat[i]*top_diff[i]' for i = 1:n_input])
  @test all(-eps .< vec(weight_grad) - vec(weight_grad_expected) .< eps)

  back_grad = [similar(X_mat[i]) for i = 1:n_input]
  for i = 1:n_input
    copy!(back_grad[i], diffs[i])
    @test all(-eps .< vec(back_grad[i]) - vec(W * top_diff[i]) .< eps)
  end

  shutdown(backend, state)
end
function test_inner_product_layer(backend::Backend)
  test_inner_product_layer(backend, 3, Float32, 1e-3)
  test_inner_product_layer(backend, 3, Float64, 1e-10)
end

if test_cpu
  test_inner_product_layer(backend_cpu)
end
if test_gpu
  test_inner_product_layer(backend_gpu)
end
