function test_softmax_layer(sys::System)
  println("-- Testing SoftmaxLayer on $(typeof(sys.backend))...")

  eps = 1e-10
  width, height, channels, num = (5, 6, 7, 8)
  input = rand(width, height, channels, num)
  input_blob = make_blob(sys.backend, input)

  layer = SoftmaxLayer(tops = String["prob"], bottoms = String["response"])
  state = setup(sys, layer, Blob[input_blob])

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

  got_output = zeros(size(output))
  copy!(got_output, state.blobs[1])

  @test all(-eps .< output - got_output .< eps)
end

if test_cpu
  test_softmax_layer(sys_cpu)
end
if test_cudnn
  test_softmax_layer(sys_cudnn)
end
