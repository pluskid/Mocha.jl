function test_multinomial_logistic_loss_layer(sys::System, class_weights, T, eps)
  println("-- Testing MultinomialLogisticLossLayer{$(class_weights[1]),$(class_weights[2])} on $(typeof(sys.backend)){$T}...")

  width, height, channels, num = (5, 6, 7, 8)

  prob = abs(rand(T, width, height, channels, num))
  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{T}, label)

  prob_blob = make_blob(sys.backend, prob)
  label_blob = make_blob(sys.backend, label)
  inputs = Blob[prob_blob, label_blob]

  if class_weights[1] == :equal
    weights = ones(T, channels)
  elseif class_weights[1] == :local
    weights = rand(T, channels)
  elseif class_weights[1] == :global
    weights = rand(T, width, height, channels)
  else
    @assert class_weights[1] == :no
    weights = []
  end

  layer = MultinomialLogisticLossLayer(bottoms=[:pred, :labels],
      weights=weights, normalize=class_weights[2])
  state = setup(sys, layer, inputs, Blob[])

  forward(sys, state, inputs)

  if class_weights[1] == :local || class_weights[1] == :equal
    weights = repeat(reshape(weights,1,1,channels), inner=[width,height,1])
  end
  if class_weights[2] == :local
    weights = weights .* (channels ./ sum(weights,3))
  elseif class_weights[2] == :global
    weights = weights * (width*height*channels / sum(weights))
  else
    @assert class_weights[2] == :no
  end

  expected_loss = convert(T, 0)
  for w = 1:width
    for h = 1:height
      for n = 1:num
        if isempty(weights)
          expected_loss += -log(prob[w, h, int(label[w, h, 1, n])+1, n])
        else
          idx = int(label[w,h,1,n])+1
          expected_loss += -log(prob[w,h,idx,n] * weights[w,h,idx])
        end
      end
    end
  end
  expected_loss /= (width*height*num)

  @test -eps < state.loss - expected_loss < eps

  shutdown(sys, state)
end
function test_multinomial_logistic_loss_layer(sys::System, T, eps)
  for class_weights in ((:equal,:local),(:local,:local),(:global,:global),(:global,:local),(:no,:no))
    test_multinomial_logistic_loss_layer(sys, class_weights, T, eps)
  end
end
function test_multinomial_logistic_loss_layer(sys::System)
  test_multinomial_logistic_loss_layer(sys, Float32, 1e-3)
  test_multinomial_logistic_loss_layer(sys, Float64, 1e-5)
end

if test_cpu
  test_multinomial_logistic_loss_layer(sys_cpu)
end
if test_cudnn
  test_multinomial_logistic_loss_layer(sys_cudnn)
end

