function test_power_layer(backend::Backend, scale, shift, power, T, eps)
  println("    > scale=$scale, shift=$shift, power=$power")

  width, height, channels, num = (5, 6, 7, 8)
  input = rand(T, width, height, channels, num)
  input_blob = make_blob(backend, input)
  grad_blob = make_blob(backend, eltype(input), size(input))

  layer = PowerLayer(tops = [:prob], bottoms = [:response],
      scale=scale, shift=shift, power=power)
  state = setup(backend, layer, Blob[input_blob], Blob[grad_blob])

  forward(backend, state, Blob[input_blob])

  output = (scale * input + shift) .^ power
  got_output = zeros(T, size(output))
  copy!(got_output, state.blobs[1])

  @test all(-eps .< output - got_output .< eps)

  top_diff = rand(T, size(input))
  copy!(state.blobs_diff[1], top_diff)

  backward(backend, state, Blob[input_blob], Blob[grad_blob])

  grad = power * scale * (scale * input + shift) .^ (power - 1) .* top_diff
  got_grad = zeros(T, size(grad))
  copy!(got_grad, grad_blob)
  @test all(-eps .< got_grad - grad .< eps)

  shutdown(backend, state)
end

function simple_rand()
  int(100*rand())/100
end
function test_power_layer(backend::Backend, T, eps)
  println("-- Testing PowerLayer on $(typeof(backend)){$T}...")
  test_power_layer(backend, simple_rand(), simple_rand(), 2, T, eps)
  test_power_layer(backend, 0, simple_rand(), abs(rand(Int)) % 5 + 2, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 2, T, eps)
  test_power_layer(backend, simple_rand(), 0, 3, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 4, T, eps)

  test_power_layer(backend, simple_rand(), simple_rand(), 0, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 1, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), -1, T, eps)
end

function test_power_layer(backend::Backend)
  test_power_layer(backend, Float32, 1e-2)
  test_power_layer(backend, Float64, 1e-8)
end

if test_cpu
  test_power_layer(backend_cpu)
end
if test_gpu
  test_power_layer(backend_gpu)
end

