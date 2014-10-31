function test_hdf5_data_layer(sys::System)
  println("-- Testing InnerProductLayer...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size   = 50
  orig_dim_all = (10, 20, 3)
  orig_dim     = prod(orig_dim)
  target_dim   = 30
  eps          = 1e-10

  X = rand(orig_dim_all..., batch_size)
  W = rand(orig_dim, target_dim)
  b = rand(target_dim)

  ############################################################
  # Setup
  ############################################################
  layer  = InnerProductLayer(; output_dim=right_dim, tops = String["result"], bottoms=String["input"])
  if isa(sys.backend, CPUBackend)
    error("TODO")
  elseif isa(sys.backend, CuDNNBackend)
    inputs = Blob[Mocha.cudnn_make_pod_blob(Float64, orig_dim_all..., batch_size)]
  end
  state  = setup(sys, layer, inputs)

  @test length(state.W) == length(W)
  @test length(state.b) == length(b)
  copy!(state.W, W)
  copy!(state.b, b)

  forward(system, state, inputs)

  X2 = reshape(X, orig_dim, batch_size)
  res = W' * X .+ b

  res_layer = similar(res)
  copy!(res_layer, state.blobs[1])

  @test all(-eps .< res_layer - res .< eps)
end

test_hdf5_data_layer(sys_cudnn)
