function test_accuracy_layer(backend::Backend, T)
  println("-- Testing AccuracyLayer on $(typeof(backend)){$T}...")

  eps = 1e-5
  width, height, channels, num = (5, 6, 7, 8)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(backend, input)

  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{T}, label)
  label_blob = make_blob(backend, label)

  inputs = Blob[input_blob, label_blob]

  layer = AccuracyLayer(bottoms=[:pred, :labels])
  state = setup(backend, layer, inputs, Blob[])

  println("    > Forward")
  forward(backend, state, inputs)

  @test state.n_accum == width*height*num

  expected_acc = 0.0
  for n = 1:num
    for w = 1:width
      for h = 1:height
        pred = input[w, h, :, n]
        if indmax(pred) == convert(Int, label[w,h,1,n])+1
          expected_acc += 1
        end
      end
    end
  end
  expected_acc /= (width*height*num)
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again")
  forward(backend, state, inputs)
  @test state.n_accum == 2*width*height*num
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again and Again")
  reset_statistics(state)
  forward(backend, state, inputs)
  @test state.n_accum == width*height*num
  @test abs(state.accuracy - expected_acc) < eps

  shutdown(backend, state)
end
function test_accuracy_layer(backend::Backend)
  test_accuracy_layer(backend, Float32)
  test_accuracy_layer(backend, Float64)
end

if test_cpu
  test_accuracy_layer(backend_cpu)
end
if test_gpu
  test_accuracy_layer(backend_gpu)
end
