function test_softmax_loss_layer(sys::System, T, eps)
  println("-- Testing SoftmaxLossLayer on $(typeof(sys.backend))...")

  width, height, channels, num = (5, 6, 7, 8)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(sys.backend, input)
  diff_blob = make_blob(sys.backend, T, size(input))

  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{T}, label)
  label_blob = make_blob(sys.backend, label)

  inputs = Blob[input_blob, label_blob]

  layer = SoftmaxLossLayer(bottoms=[:pred, :labels])
  state = setup(sys, layer, inputs, Blob[diff_blob])

  println("    > Forward")
  forward(sys, state, inputs)

  expected_loss = 0
  expected_grad = zeros(T, size(input))
  for w = 1:width
    for h = 1:height
      for n = 1:num
        pred = exp(input[w, h, :, n])
        pred /= sum(pred)
        expected_grad[w, h, :, n] = pred
        expected_grad[w, h, int(label[w,h,1,n])+1, n] -= 1
        expected_loss += -log(pred[int(label[w,h,1,n])+1])
      end
    end
  end
  expected_loss /= (width*height*num)
  expected_grad /= (width*height*num)

  #println("loss = $(state.loss)")
  #println("expected_loss = $expected_loss")
  @test -eps < state.loss - expected_loss < eps

  println("    > Backward")
  backward(sys, state, inputs, Blob[diff_blob])
  grad = zeros(T, size(input))
  copy!(grad, diff_blob)

  @test all(-eps .< grad - expected_grad .< eps)

  shutdown(sys, state)
end
function test_softmax_loss_layer(sys::System)
  test_softmax_loss_layer(sys, Float32, 1e-3)
  test_softmax_loss_layer(sys, Float64, 1e-5)
end

if test_cpu
  test_softmax_loss_layer(sys_cpu)
end
if test_cudnn
  test_softmax_loss_layer(sys_cudnn)
end
