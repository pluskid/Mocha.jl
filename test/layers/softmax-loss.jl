function test_softmax_loss_layer(backend::Backend, use_weights::Bool, T, eps)
  println("-- Testing SoftmaxLossLayer on $(typeof(backend)){$T} $(use_weights ? "(with weights)" : "")...")

  width, height, channels, num = (5, 6, 7, 8)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(backend, input)
  diff_blob = make_blob(backend, T, size(input))

  if use_weights
    weights = rand(T, width, height, channels) + 0.1
    weights = weights .* (channels ./ sum(weights,3))
  else
    weights = []
  end

  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{T}, label)
  label_blob = make_blob(backend, label)

  inputs = Blob[input_blob, label_blob]

  layer = SoftmaxLossLayer(bottoms=[:pred, :labels], weights=weights, normalize=:local)
  state = setup(backend, layer, inputs, Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, inputs)

  expected_loss = 0
  expected_grad = zeros(T, size(input))
  for w = 1:width
    for h = 1:height
      for n = 1:num
        pred = exp(input[w, h, :, n])
        pred /= sum(pred)
        if isempty(weights)
          expected_grad[w, h, :, n] = pred
          expected_grad[w, h, int(label[w,h,1,n])+1, n] -= 1
          expected_loss += -log(pred[int(label[w,h,1,n])+1])
        else
          y = int(label[w,h,1,n])+1
          expected_grad[w, h, :, n] = pred .* weights[w, h, y]
          expected_grad[w, h, y, n] -= weights[w,h,y]
          expected_loss += -log(pred[y]) * weights[w,h,y]
        end
      end
    end
  end
  expected_loss /= (width*height*num)
  expected_grad /= (width*height*num)

  #println("loss = $(state.loss)")
  #println("expected_loss = $expected_loss")
  @test -eps < state.loss - expected_loss < eps

  println("    > Backward")
  backward(backend, state, inputs, Blob[diff_blob])
  grad = zeros(T, size(input))
  copy!(grad, diff_blob)

  @test all(-eps .< grad - expected_grad .< eps)

  shutdown(backend, state)
end
function test_softmax_loss_layer(backend::Backend, T, eps)
  test_softmax_loss_layer(backend, false, T, eps)
  test_softmax_loss_layer(backend, true, T, eps)
end
function test_softmax_loss_layer(backend::Backend)
  test_softmax_loss_layer(backend, Float64, 1e-5)
  test_softmax_loss_layer(backend, Float32, 1e-3)
end

if test_cpu
  test_softmax_loss_layer(backend_cpu)
end
if test_cudnn
  test_softmax_loss_layer(backend_cudnn)
end
