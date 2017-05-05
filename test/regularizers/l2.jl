function test_l2_regularizer(backend::Backend, T, eps)
  println("-- Testing L2 regularizer on $(typeof(backend)){$T}...")

  coef = rand()
  param = rand(T, 2,3,4,5) - convert(T, 0.5)
  param_blob = make_blob(backend, param)
  regu = L2Regu(coef)

  loss = forward(backend, regu, 1.0, param_blob)
  expected_loss = coef * vecnorm(param)^2
  @test -eps < loss - expected_loss < eps

  grad_blob = make_zero_blob(backend, T, size(param))
  backward(backend, regu, 1.0, param_blob, grad_blob)
  grad = zeros(T, size(param))
  copy!(grad, grad_blob)

  @test all(-eps .< grad - 2coef*param .< eps)
end

function test_l2_regularizer(backend::Backend)
  test_l2_regularizer(backend, Float32, 1e-5)
  test_l2_regularizer(backend, Float64, 1e-10)
end

if test_cpu
  test_l2_regularizer(backend_cpu)
end
if test_gpu
  test_l2_regularizer(backend_gpu)
end

