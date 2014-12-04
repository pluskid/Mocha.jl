function test_reshape_layer(backend::Backend, T, eps)
  println("-- Testing ReshapeLayer on $(typeof(backend)){$T}...")

  width, height, channels, num = (3,4,5,6)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(backend, input)
  diff_blob  = make_blob(backend, input)

  println("    > Setup")
  layer = ReshapeLayer(bottoms=[:input], tops=[:output],
      width=1, height=1, channels=width*height*channels)
  state = setup(backend, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(input))
  copy!(got_output, input_blob)
  @test all(abs(got_output - input) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(input))
  copy!(diff_blob, top_diff)
  backward(backend, state, Blob[input_blob], Blob[diff_blob])
  got_grad = zeros(T, size(input))
  copy!(got_grad, diff_blob)
  @test all(abs(got_grad - top_diff) .< eps)

  shutdown(backend, state)
end
function test_reshape_layer(backend::Backend)
  test_reshape_layer(backend, Float64, 1e-10)
  test_reshape_layer(backend, Float32, 1e-4)
end

if test_cpu
  test_reshape_layer(backend_cpu)
end
if test_cudnn
  test_reshape_layer(backend_cudnn)
end
