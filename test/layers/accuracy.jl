function test_accuracy_layer(sys::System)
  println("-- Testing AccuracyLayer on $(typeof(sys.backend))...")

  eps = 1e-5
  width, height, channels, num = (5, 6, 7, 8)
  input = rand(width, height, channels, num)
  input_blob = make_blob(sys.backend, input)

  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{Float64}, label)
  label_blob = make_blob(sys.backend, label)

  inputs = Blob[input_blob, label_blob]

  layer = AccuracyLayer(bottoms=[:pred, :labels])
  state = setup(sys, layer, inputs, Blob[])

  println("    > Forward")
  forward(sys, state, inputs)

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
  forward(sys, state, inputs)
  @test state.n_accum == 2*width*height*num
  @test abs(state.accuracy - expected_acc) < eps

  println("    > Forward Again and Again")
  reset_statistics(state)
  forward(sys, state, inputs)
  @test state.n_accum == width*height*num
  @test abs(state.accuracy - expected_acc) < eps

  shutdown(sys, state)
end

if test_cpu
  test_accuracy_layer(sys_cpu)
end
if test_cudnn
  test_accuracy_layer(sys_cudnn)
end
