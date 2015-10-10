function test_power_layer(backend::Backend, scale, shift, power, n_input, T, eps)
  tensor_dim = abs(rand(Int)) % 6 + 1
  println("    > scale=$scale, shift=$shift, power=$power, tensor_dim=$tensor_dim")

  dims = [abs(rand(Int, tensor_dim)) % 5 + 5 for i = 1:n_input]
  input = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  grad_blob = Blob[make_blob(backend, x) for x in input]

  layer = PowerLayer(tops=Array(Symbol,n_input), bottoms=Array(Symbol,n_input),
      scale=scale, shift=shift, power=power)
  state = setup(backend, layer, input_blob, grad_blob)

  forward(backend, state, input_blob)

  for i = 1:n_input
    output = (scale * input[i] + shift) .^ power
    got_output = zeros(T, size(output))
    copy!(got_output, state.blobs[i])

    @test all(-eps .< output - got_output .< eps)
  end

  top_diff = [rand(T, size(input[i])) for i = 1:n_input]
  for i = 1:n_input
    copy!(state.blobs_diff[i], top_diff[i])
  end

  backward(backend, state, input_blob, grad_blob)

  for i = 1:n_input
    grad = power * scale * (scale * input[i] + shift) .^ (power - 1) .* top_diff[i]
    got_grad = zeros(T, size(grad))
    copy!(got_grad, grad_blob[i])
    @test all(-eps .< got_grad - grad .< eps)
  end

  shutdown(backend, state)
end

function simple_rand()
  round(Int, 100*rand())/100
end
function test_power_layer(backend::Backend, n_input, T, eps)
  println("-- Testing PowerLayer on $(typeof(backend)){$T}...")
  test_power_layer(backend, simple_rand(), simple_rand(), 2, n_input, T, eps)
  test_power_layer(backend, 0, simple_rand(), abs(rand(Int)) % 5 + 2, n_input, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 2, n_input, T, eps)
  test_power_layer(backend, simple_rand(), 0, 3, n_input, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 4, n_input, T, eps)

  test_power_layer(backend, simple_rand(), simple_rand(), 0, n_input, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), 1, n_input, T, eps)
  test_power_layer(backend, simple_rand(), simple_rand(), -1, n_input, T, eps)
end

function test_power_layer(backend::Backend)
  test_power_layer(backend, 3, Float32, 8e-2)
  test_power_layer(backend, 3, Float64, 1e-8)
end

if test_cpu
  test_power_layer(backend_cpu)
end
if test_gpu
  test_power_layer(backend_gpu)
end

