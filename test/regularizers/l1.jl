function test_l1_regularizer(backend::Backend, T, eps)
  println("-- Testing L1 regularizer on $(typeof(backend)){$T}...")

  coef = rand()
  param = rand(T, 2,3,4,5) - 0.5
  param_blob = make_blob(backend, param)
  regu = L1Regu(coef)

  loss = forward(backend, regu, 1.0, param_blob)
  expected_loss = coef * sum(abs(param))
  @test -eps < loss - expected_loss < eps

  grad_blob = make_zero_blob(backend, T, size(param))
  backward(backend, regu, 1.0, param_blob, grad_blob)
  grad = zeros(T, size(param))
  copy!(grad, grad_blob)

  @test all(-eps .< grad - coef*sign(param) .< eps)
end

function test_l1_regularizer(backend::Backend)
  test_l1_regularizer(backend, Float32, 1e-3)
  test_l1_regularizer(backend, Float64, 1e-4)
end

if test_cpu
  test_l1_regularizer(backend_cpu)
end
if test_gpu
  test_l1_regularizer(backend_gpu)
end

