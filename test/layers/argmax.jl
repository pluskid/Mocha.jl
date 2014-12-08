function test_argmax_layer(backend::Backend, n_input, T, eps)
  println("-- Testing ArgmaxLayer on $(typeof(backend)){$T}...")

  dims = [abs(rand(Int, 4)) % 6 + 1 for i = 1:n_input]
  input = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  println("    > Setup")
  layer = ArgmaxLayer(bottoms=Array(Symbol,n_input),tops=Array(Symbol,n_input))
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)
  for i = 1:n_input
    width,height,channels,num = dims[i]
    got_output = zeros(T, width, height, 1, num)
    expected_output = similar(got_output)
    for n = 1:num
      for w = 1:width
        for h = 1:height
          expected_output[w,h,1,n] = indmax(input[i][w,h,:,n])-1
        end
      end
    end

    copy!(got_output, state.blobs[i])
    @test all(abs(got_output - expected_output) .< eps)
  end

  shutdown(backend, state)
end
function test_argmax_layer(backend::Backend)
  test_argmax_layer(backend, 3, Float64, 1e-10)
  test_argmax_layer(backend, 3, Float32, 1e-10)
end

if test_cpu
  test_argmax_layer(backend_cpu)
end
if test_gpu
  test_argmax_layer(backend_gpu)
end

