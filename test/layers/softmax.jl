function test_softmax_layer(backend::Backend, tensor_dim, n_input, T, eps)
  println("-- Testing SoftmaxLayer on $(typeof(backend)){$T}...")

  norm_dim = max(1, abs(rand(Int)) % tensor_dim)
  println("    > $tensor_dim-dimensional input, normalize along dimension $norm_dim")

  dims = [rand(6:11, tensor_dim) for i = 1:n_input]
  input = [rand(T, dims[i]...) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob = Blob[make_blob(backend, x) for x in input]

  layer = SoftmaxLayer(tops=Array{Symbol}(n_input), bottoms=Array{Symbol}(n_input),
      dim=norm_dim-tensor_dim-1)
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)

  for i = 1:n_input
    my_dims  = size(input[i])
    dim_pre  = prod(my_dims[1:norm_dim-1])
    dim_prob = my_dims[norm_dim]
    dim_post = prod(my_dims[norm_dim+1:end])

    canonical_input = reshape(input[i], (dim_pre, dim_prob, dim_post))
    output = similar(canonical_input)

    for x = 1:dim_pre
      for y = 1:dim_post
        preds = canonical_input[x,:,y]
        preds -= maximum(preds)
        preds = exp.(preds)
        preds /= sum(preds)
        output[x,:,y] = preds
      end
    end
    output = reshape(output, my_dims)

    got_output = zeros(T, size(output))
    copy!(got_output, state.blobs[i])

    @test all(-eps .< output[:] - got_output[:] .< eps)
  end

  println("    > Backward")
  for i = 1:n_input
    copy!(state.blobs_diff[i], rand(T, size(input[i])))
  end
  backward(backend, state, input_blob, diff_blob)

  for i = 1:n_input
    my_dims  = size(input[i])
    dim_pre  = prod(my_dims[1:norm_dim-1])
    dim_prob = my_dims[norm_dim]
    dim_post = prod(my_dims[norm_dim+1:end])

    canonical_output = reshape(to_array(state.blobs[i]), (dim_pre, dim_prob, dim_post))
    canonical_topdiff = reshape(to_array(state.blobs_diff[i]), (dim_pre, dim_prob, dim_post))
    grad = similar(canonical_output)

    for x = 1:dim_pre
      for y = 1:dim_post
        topdiff0 = canonical_topdiff[x,:,y]
        output0 = canonical_output[x,:,y]
        grad[x,:,y] = topdiff0.*output0 - dot(vec(topdiff0), vec(output0))*output0
      end
    end

    got_grad = to_array(diff_blob[i])
    @test all(-eps .< vec(got_grad)-vec(grad) .< eps)
  end

  shutdown(backend, state)
end
function test_softmax_layer(backend::Backend, n_input, T, eps)
  for td in [2,4,5]
    test_softmax_layer(backend, td, n_input, T, eps)
  end
end
function test_softmax_layer(backend::Backend)
  test_softmax_layer(backend, 3, Float64, 1e-10)
  test_softmax_layer(backend, 3, Float32, 1e-5)
end

if test_gpu
  test_softmax_layer(backend_gpu)
end
if test_cpu
  test_softmax_layer(backend_cpu)
end
