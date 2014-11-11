function test_l2_regularizer(sys::System)
  println("-- Testing L2 regularizer on $(typeof(sys.backend))...")

  eps = 1e-10
  param = rand(2,3,4,5) - 0.5
  param_blob = make_blob(sys.backend, param)
  regu = L2Regu(1)

  loss = forward(sys, regu, param_blob)
  expected_loss = vecnorm(param)^2
  @test -eps < loss - expected_loss < eps

  grad_blob = make_zero_blob(sys.backend, Float64, size(param))
  backward(sys, regu, param_blob, grad_blob)
  grad = zeros(size(param))
  copy!(grad, grad_blob)

  @test all(-eps .< grad - param .< eps)
end

if test_cpu
  test_l2_regularizer(sys_cpu)
end
if test_cudnn
  test_l2_regularizer(sys_cudnn)
end

