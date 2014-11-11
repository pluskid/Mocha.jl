function test_multinomial_logistic_loss_layer(sys::System)
  println("-- Testing MultinomialLogisticLossLayer on $(typeof(sys.backend))...")

  eps = 1e-5
  width, height, channels, num = (5, 6, 7, 8)

  prob = abs(rand(width, height, channels, num))
  label = abs(rand(Int, (width, height, 1, num))) % channels
  label = convert(Array{Float64}, label)

  prob_blob = make_blob(sys.backend, prob)
  label_blob = make_blob(sys.backend, label)
  inputs = Blob[prob_blob, label_blob]

  layer = MultinomialLogisticLossLayer(bottoms=[:pred, :labels])
  state = setup(sys, layer, inputs)

  forward(sys, state, inputs)

  expected_loss = 0f0
  for w = 1:width
    for h = 1:height
      for n = 1:num
        expected_loss += -log(prob[w, h, int(label[w, h, 1, n])+1, n])
      end
    end
  end
  expected_loss /= (width*height*num)

  #println("loss = $(state.loss)")
  #println("expected_loss = $expected_loss")
  @test -eps < state.loss - expected_loss < eps
end

if test_cpu
  test_multinomial_logistic_loss_layer(sys_cpu)
end
if test_cudnn
  test_multinomial_logistic_loss_layer(sys_cudnn)
end

