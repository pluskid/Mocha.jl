function test_softmax_loss_layer(backend::Backend, use_weights::Bool, T, eps)
  println("-- Testing SoftmaxLossLayer on $(typeof(backend)){$T} $(use_weights ? "(with weights)" : "")...")

  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple((abs(rand(Int,tensor_dim)) % 6 + 6)...)
  println("    > $dims")

  input = rand(T, dims)
  width, height, channels, num = get_whcn(input)

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
  canonical_input = reshape(input, (width,height,channels,num))
  canonical_grad = reshape(expected_grad, (width,height,channels,num))
  for w = 1:width
    for h = 1:height
      for n = 1:num
        pred = exp(canonical_input[w, h, :, n])
        pred /= sum(pred)
        if isempty(weights)
          canonical_grad[w, h, :, n] = pred
          canonical_grad[w, h, int(label[w,h,1,n])+1, n] -= 1
          expected_loss += -log(pred[int(label[w,h,1,n])+1])
        else
          y = int(label[w,h,1,n])+1
          canonical_grad[w, h, :, n] = pred .* weights[w,h,y]
          canonical_grad[w, h, y, n] -= weights[w,h,y]
          expected_loss += -log(pred[y]) * weights[w,h,y]
        end
      end
    end
  end
  expected_loss /= (width*height*num)
  expected_grad /= (width*height*num)

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
if test_gpu
  test_softmax_loss_layer(backend_gpu)
end
