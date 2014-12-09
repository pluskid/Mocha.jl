function test_softmax_layer(backend::Backend, n_input, T, eps)
  println("-- Testing SoftmaxLayer on $(typeof(backend)){$T}...")

  dims = [abs(rand(Int,4)) % 6 + 6 for i = 1:n_input]
  input = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  layer = SoftmaxLayer(tops=Array(Symbol,n_input), bottoms=Array(Symbol,n_input))
  state = setup(backend, layer, input_blob, diff_blob)

  forward(backend, state, input_blob)

  for i = 1:n_input
    output = similar(input[i])
    width, height, channels, num = dims[i]

    for w = 1:width
      for h = 1:height
        for n = 1:num
          preds = input[i][w, h, :, n]
          preds -= maximum(preds)
          preds = exp(preds)
          preds /= sum(preds)
          output[w, h, :, n] = preds
        end
      end
    end

    got_output = zeros(T, size(output))
    copy!(got_output, state.blobs[i])

    @test all(-eps .< output - got_output .< eps)
  end

  shutdown(backend, state)
end
function test_softmax_layer(backend::Backend)
  test_softmax_layer(backend, 3, Float32, 1e-5)
  test_softmax_layer(backend, 3, Float64, 1e-10)
end

if test_cpu
  test_softmax_layer(backend_cpu)
end
if test_gpu
  test_softmax_layer(backend_gpu)
end
