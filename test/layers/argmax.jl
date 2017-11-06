function test_argmax_layer(backend::Backend, n_input, tensor_dim, T, eps)
  println("-- Testing ArgmaxLayer on $(typeof(backend)){$T}...")

  println("    > $tensor_dim-dimensional tensor")

  dims = [rand(1:6, tensor_dim) for i = 1:n_input]
  op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  inputs = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in inputs]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  println("    > Setup")
  layer = ArgmaxLayer(bottoms=Array{Symbol}(n_input),tops=Array{Symbol}(n_input),dim=op_dim)
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)
  for i = 1:n_input
    outdim = [size(inputs[i])...]
    outdim[op_dim] = 1
    got_output = zeros(T, outdim...)
    expected_output = zeros(T, outdim...)

    pre_dim, mid_dim, post_dim = split_dims(inputs[i], op_dim)
    input = reshape(inputs[i], pre_dim, mid_dim, post_dim)
    output = reshape(expected_output, pre_dim, 1, post_dim)
    for x = 1:pre_dim
      for z = 1:post_dim
        output[x,1,z] = indmax(input[x,:,z])-1
      end
    end

    copy!(got_output, state.blobs[i])
    @test all(abs.(got_output - expected_output) .< eps)
  end

  shutdown(backend, state)
end
function test_argmax_layer(backend::Backend, n_input, T, eps)
  for i in [2,4,5]
    test_argmax_layer(backend, n_input, i, T, eps)
  end
end
function test_argmax_layer(backend::Backend)
  test_argmax_layer(backend, 3, Float64, 1e-10)
  test_argmax_layer(backend, 3, Float32, 1e-10)
end

if test_cpu
  test_argmax_layer(backend_cpu)
end
if test_gpu
  test_argmax_layer(backend_gpu)
end

