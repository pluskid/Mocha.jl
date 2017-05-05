function test_l2_constraint(backend::Backend, T, eps)
  println("-- Testing L2 constraint on $(typeof(backend)){$T}...")
  # this simulates a convolutional filter and applies
  # the l2 constraint to it
  n_filters = 5
  coef = 0.2
  param = rand(T, 2,3,4,n_filters) - convert(T, 0.5)
  param_after = zeros(T, size(param))
  param_blob = make_blob(backend, param)

  cons = L2Cons(coef)
  constrain!(backend, cons, param_blob)
  copy!(param_after, param_blob)
  param_after = reshape(param_after, size(param))
  for f=1:n_filters
    norm2 = vecnorm(param_after[:, :, :, f])
    @test norm2 <= coef + eps
  end

  # this is the same as above but for fully connected weights
  n_input = 10
  n_out   = 12
  param = rand(T, n_input,n_out) - convert(T, 0.5)
  param_after = zeros(T, size(param))
  param_blob = make_blob(backend, param)

  cons = L2Cons(coef)
  constrain!(backend, cons, param_blob)
  copy!(param_after, param_blob)
  param_after = reshape(param_after, size(param))
  for f=1:n_out
    norm2 = vecnorm(param_after[:, f])
    @test norm2 <= coef + eps
  end

  # The case for bias
  len = 10
  param = rand(T, len) - convert(T, 0.5)
  param_after = similar(param)
  param_blob = make_blob(backend, param)
  cons = L2Cons(coef)
  constrain!(backend, cons, param_blob)
  copy!(param_after, param_blob)
  @test vecnorm(param_after) <= coef+eps
end

function test_l2_constraint(backend::Backend)
  test_l2_constraint(backend, Float32, 1e-5)
  test_l2_constraint(backend, Float64, 1e-5)
end

if test_cpu
  test_l2_constraint(backend_cpu)
end
if test_gpu
  test_l2_constraint(backend_gpu)
end

