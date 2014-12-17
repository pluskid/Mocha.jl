function test_reshape_layer(backend::Backend, n_input, T, eps)
  println("-- Testing ReshapeLayer on $(typeof(backend)){$T}...")

  # make sure the prod of the first 3 dimensions are the same
  dims = [(3*2^isodd(i), 2*2^iseven(i), 4, abs(rand(Int))%5+3) for i = 1:n_input]

  input = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob  = Blob[make_blob(backend, x) for x in input]

  println("    > Setup")
  layer = ReshapeLayer(bottoms=Array(Symbol,n_input), tops=Array(Symbol,n_input),
      shape=(1,1,prod(dims[1][1:end-1])))
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)
  for i = 1:n_input
    got_output = zeros(T, size(input[i]))
    copy!(got_output, input_blob[i])
    @test all(abs(got_output - input[i]) .< eps)
  end

  println("    > Backward")
  top_diff = [rand(T, size(input[i])) for i = 1:n_input]
  for i = 1:n_input
    copy!(diff_blob[i], top_diff[i])
  end
  backward(backend, state, input_blob, diff_blob)
  for i = 1:n_input
    got_grad = zeros(T, size(input[i]))
    copy!(got_grad, diff_blob[i])
    @test all(abs(got_grad - top_diff[i]) .< eps)
  end

  shutdown(backend, state)
end
function test_reshape_layer(backend::Backend)
  test_reshape_layer(backend, 3, Float64, 1e-10)
  test_reshape_layer(backend, 3, Float32, 1e-4)
end

if test_cpu
  test_reshape_layer(backend_cpu)
end
if test_gpu
  test_reshape_layer(backend_gpu)
end
