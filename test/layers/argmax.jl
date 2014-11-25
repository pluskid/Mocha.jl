function test_argmax_layer(sys::System, T, eps)
  println("-- Testing ArgmaxLayer on $(typeof(sys.backend)){$T}...")

  width, height, channels, num = (3,4,5,6)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(sys.backend, input)

  println("    > Setup")
  layer = ArgmaxLayer(bottoms=[:input], tops=[:output])
  state = setup(sys, layer, Blob[input_blob], Blob[NullBlob()])

  println("    > Forward")
  forward(sys, state, Blob[input_blob])
  got_output = zeros(T, width, height, 1, num)
  expected_output = similar(got_output)
  for n = 1:num
    for w = 1:width
      for h = 1:height
        expected_output[w,h,1,n] = indmax(input[w,h,:,n])-1
      end
    end
  end

  copy!(got_output, state.blobs[1])
  @test all(abs(got_output - expected_output) .< eps)

  shutdown(sys, state)
end
function test_argmax_layer(sys::System)
  test_argmax_layer(sys, Float64, 1e-10)
  test_argmax_layer(sys, Float32, 1e-10)
end

if test_cpu
  test_argmax_layer(sys_cpu)
end
if test_cudnn
  test_argmax_layer(sys_cudnn)
end

