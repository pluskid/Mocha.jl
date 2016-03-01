function test_gaussian_kl_loss_layer(backend::Backend, T, eps)
  println("-- Testing GaussianKLLossLayer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple((abs(rand(Int,tensor_dim)) % 6 + 6)...)
  println("    > $dims")
  mus = rand(T, dims)
  sigmas = sqrt(rand(T, dims).^2)

  ############################################################
  # Setup
  ############################################################
  weight = 1.1
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
  loss = 0.5(sum(mus.^2 + sigmas.^2 - 2log(sigmas)) - n)
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

if test_cuda
  test_gaussian_kl_loss_layer(backend_cuda)
end

if test_cpu
  test_gaussian_kl_loss_layer(backend_cpu)
end

if test_opencl
  warn("TODO: OpenCL gaussian kl loss layer tests")
end
