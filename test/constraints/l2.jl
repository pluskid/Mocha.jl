function test_l2_constraint(sys::System, T, eps)
  println("-- Testing L2 constraint on $(typeof(sys.backend)){$T}...")
  # this simulates a convolutional filter and applies
  # the l2 constraint to it
  n_filters = 5
  coef = 0.2
  param = rand(T, 2,3,4,n_filters) - 0.5
  param_after = zeros(T, size(param))
  param_blob = make_blob(sys.backend, param)
 
  cons = L2Cons(coef)
  constrain!(sys, cons, param_blob)
  copy!(param_after, param_blob)
  param_after = reshape(param_after, size(param))
  for f=1:n_filters
    norm2 = vecnorm(param_after[:, :, :, f])
    @test norm2 <= coef + eps
  end

  # this is the same as above but for fully connected weights
  n_input = 10
  n_out   = 12
  param = rand(T, n_input,n_out,1,1) - 0.5
  param_after = zeros(T, size(param))
  param_blob = make_blob(sys.backend, param)
 
  cons = L2Cons(coef)
  constrain!(sys, cons, param_blob)
  copy!(param_after, param_blob)
  param_after = reshape(param_after, size(param))
  for f=1:n_out
    norm2 = vecnorm(param_after[:, f, :, :])
    @test norm2 <= coef + eps
  end
end

function test_l2_constraint(sys::System)
  test_l2_constraint(sys, Float32, 1e-5)
  test_l2_constraint(sys, Float64, 1e-5)
end

if test_cpu
  test_l2_constraint(sys_cpu)
end
if test_cudnn
  test_l2_constraint(sys_cudnn)
end

