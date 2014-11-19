function test_softmax_layer(sys::System, T, eps)
  println("-- Testing SoftmaxLayer on $(typeof(sys.backend)){$T}...")

  width, height, channels, num = (5, 6, 7, 8)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(sys.backend, input)

  layer = SoftmaxLayer(tops = [:prob], bottoms = [:response])
  state = setup(sys, layer, Blob[input_blob], Blob[])

  forward(sys, state, Blob[input_blob])

  output = similar(input)
  for w = 1:width
    for h = 1:height
      for n = 1:num
        preds = input[w, h, :, n]
        preds -= maximum(preds)
        preds = exp(preds)
        preds /= sum(preds)
        output[w, h, :, n] = preds
      end
    end
  end

  got_output = zeros(T, size(output))
  copy!(got_output, state.blobs[1])

  @test all(-eps .< output - got_output .< eps)

  shutdown(sys, state)
end
function test_softmax_layer(sys::System)
  test_softmax_layer(sys, Float32, 1e-5)
  test_softmax_layer(sys, Float64, 1e-10)
end

if test_cpu
  test_softmax_layer(sys_cpu)
end
if test_cudnn
  test_softmax_layer(sys_cudnn)
end
