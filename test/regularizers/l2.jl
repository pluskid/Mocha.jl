function test_l2_regularizer(sys::System, T)
  println("-- Testing L2 regularizer on $(typeof(sys.backend)){$T}...")

  eps = 1e-10
  param = rand(T, 2,3,4,5) - 0.5
  param_blob = make_blob(sys.backend, param)
  regu = L2Regu(1)

  loss = forward(sys, regu, 1.0, param_blob)
  expected_loss = vecnorm(param)^2
  @test -eps < loss - expected_loss < eps

  grad_blob = make_zero_blob(sys.backend, T, size(param))
  backward(sys, regu, 1.0, param_blob, grad_blob)
  grad = zeros(T, size(param))
  copy!(grad, grad_blob)

  @test all(-eps .< grad - param .< eps)
end

function test_l2_regularizer(sys::System)
  test_l2_regularizer(sys, Float32)
  test_l2_regularizer(sys, Float64)
end

if test_cpu
  test_l2_regularizer(sys_cpu)
end
if test_cudnn
  test_l2_regularizer(sys_cudnn)
end

