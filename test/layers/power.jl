function test_power_layer(sys::System, scale, shift, power)
  println("    > scale=$scale, shift=$shift, power=$power")

  eps = 1e-10
  width, height, channels, num = (5, 6, 7, 8)
  input = rand(width, height, channels, num)
  input_blob = make_blob(sys.backend, input)

  layer = PowerLayer(tops = [:prob], bottoms = [:response],
      scale=scale, shift=shift, power=power)
  state = setup(sys, layer, Blob[input_blob])

  forward(sys, state, Blob[input_blob])

  output = (scale * input + shift) .^ power
  got_output = zeros(size(output))
  copy!(got_output, state.blobs[1])

  @test all(-eps .< output - got_output .< eps)

  top_diff = rand(size(input))
  copy!(state.blobs_diff[1], top_diff)

  grad_blob = make_blob(sys.backend, top_diff)
  backward(sys, state, Blob[input_blob], Blob[grad_blob])

  grad = power * scale * (scale * input + shift) .^ (power - 1) .* top_diff
  got_grad = zeros(size(grad))
  copy!(got_grad, grad_blob)
  @test all(-eps .< got_grad - grad .< eps)
end

function test_power_layer(sys::System)
  println("-- Testing PowerLayer on $(typeof(sys.backend))...")
  test_power_layer(sys, rand(), rand(), 2)
  test_power_layer(sys, 0, rand(), abs(rand(Int)) % 5 + 2)
  test_power_layer(sys, rand(), rand(), 2)
  test_power_layer(sys, rand(), 0, 3)
  test_power_layer(sys, rand(), rand(), 4)
end

if test_cpu
  test_power_layer(sys_cpu)
end
if test_cudnn
  test_power_layer(sys_cudnn)
end

