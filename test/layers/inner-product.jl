function test_inner_product_layer(sys::System, T)
  println("-- Testing InnerProductLayer on $(typeof(sys.backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size   = 50
  orig_dim_all = (10, 20, 3)
  orig_dim     = prod(orig_dim_all)
  target_dim   = 30
  eps          = 1e-10

  X = rand(T, orig_dim_all..., batch_size)
  W = rand(T, orig_dim, target_dim)
  b = rand(T, target_dim)

  ############################################################
  # Setup
  ############################################################
  layer  = InnerProductLayer(; output_dim=target_dim, tops = [:result], bottoms=[:input])
  input_blob = make_blob(sys.backend, T, orig_dim_all..., batch_size)
  diff_blob = make_blob(sys.backend, T, size(input_blob)...)
  copy!(input_blob, X)
  inputs = Blob[input_blob]
  diffs = Blob[diff_blob]

  println("    > Setup")
  state  = setup(sys, layer, inputs, diffs)

  @test length(state.W) == length(W)
  @test length(state.b) == length(b)
  copy!(state.W, W)
  copy!(state.b, b)

  println("    > Forward")
  forward(sys, state, inputs)

  X2 = reshape(X, orig_dim, batch_size)
  res = W' * X2 .+ b

  res_layer = similar(res)
  copy!(res_layer, state.blobs[1])

  @test all(-eps .< res_layer - res .< eps)

  println("    > Backward")
  top_diff = rand(T, size(state.blobs_diff[1]))
  copy!(state.blobs_diff[1], top_diff)
  backward(sys, state, inputs, diffs)

  top_diff = reshape(top_diff, target_dim, batch_size)
  bias_grad = similar(b)
  copy!(bias_grad, state.∇b)
  @test all(-eps .< vec(bias_grad) - vec(sum(top_diff,2)) .< eps)

  X_mat = reshape(X, orig_dim, batch_size)
  weight_grad = similar(W)
  copy!(weight_grad, state.∇W)
  @test all(-eps .< vec(weight_grad) - vec(X_mat*top_diff') .< eps)

  back_grad = similar(X_mat)
  copy!(back_grad, diffs[1])
  @test all(-eps .< vec(back_grad) - vec(W * top_diff) .< eps)

  shutdown(sys, state)
end
function test_inner_product_layer(sys::System)
  test_inner_product_layer(sys, Float32)
  test_inner_product_layer(sys, Float64)
end

if test_cpu
  test_inner_product_layer(sys_cpu)
end
if test_cudnn
  test_inner_product_layer(sys_cudnn)
end
