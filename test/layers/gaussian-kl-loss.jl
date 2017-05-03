function test_gaussian_kl_loss_layer(backend::Backend, T, eps)
  println("-- Testing GaussianKLLossLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  tensor_dim = rand(2:5)
  dims = tuple(rand(6:11, tensor_dim)...)
  println("    > $dims")
  mus = rand(T, dims)
  sigmas = sqrt.(rand(T, dims).^2)

  ############################################################
  # Setup
  ############################################################
  weight = T(1.1)
  layer  = GaussianKLLossLayer(; bottoms=[:predictions, :labels], weight=weight)
  mu_blob  = make_blob(backend, T, dims)
  sigma_blob = make_blob(backend, T, dims)

  mu_diff_blob  = make_blob(backend, T, dims)
  sigma_diff_blob  = make_blob(backend, T, dims)

  copy!(mu_blob, mus)
  copy!(sigma_blob, sigmas)
  inputs = Blob[mu_blob, sigma_blob]
  diffs = Blob[mu_diff_blob, sigma_diff_blob]

  state  = setup(backend, layer, inputs, diffs)

  forward(backend, state, inputs)

  n = length(mu_blob)
  loss = 0.5(sum(mus.^2 + sigmas.^2 - 2log.(sigmas)) - n)
  loss *= weight/get_num(mu_blob)
  @test -eps < loss-state.loss < eps


  backward(backend, state, inputs, diffs)
  grad = mus
  grad *= weight/get_num(mu_blob)
  diff = similar(grad)
  copy!(diff, diffs[1])
  @test all(-eps .< grad - diff .< eps)

  grad = sigmas - 1./sigmas
  grad *= weight/get_num(mu_blob)
  diff = similar(grad)
  copy!(diff, diffs[2])
  @test all(-eps .< grad - diff .< eps)

  shutdown(backend, state)
end

function test_gaussian_kl_loss_layer(backend::Backend)
  test_gaussian_kl_loss_layer(backend, Float32, 5e-2)
  test_gaussian_kl_loss_layer(backend, Float64, 1e-8)
end

if test_gpu
  test_gaussian_kl_loss_layer(backend_gpu)
end

if test_cpu
  test_gaussian_kl_loss_layer(backend_cpu)
end
