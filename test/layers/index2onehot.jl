function test_index2onehot_layer(backend::Backend, tensor_dim, n_input, T, eps)
  println("-- Testing Index2OnehotLayer on $(typeof(backend)){$T}...")

  expand_dim = max(1, abs(rand(Int)) % tensor_dim)
  println("    > $tensor_dim-dimensional input, expanding along dimension $expand_dim")
  dims = [rand(6:11, tensor_dim) for i in 1:n_input]
  for i = 1:n_input
    dims[i][expand_dim] = 1
  end
  n_class = 6
  input = [convert(Array{T}, abs.(rand(Int, dims[i]...)) .% n_class) for i = 1:n_input]
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  layer = Index2OnehotLayer(tops=Array{Symbol}(n_input), bottoms=Array{Symbol}(n_input),
      dim=expand_dim, n_class=n_class)
  state = setup(backend, layer, input_blob, diff_blob)

  forward(backend, state, input_blob)

  for i = 1:n_input
    my_dims  = size(input[i])
    dim_pre  = prod(my_dims[1:expand_dim-1])
    dim_post = prod(my_dims[expand_dim+1:end])
    canonical_input = reshape(input[i], (dim_pre, dim_post))
    canonical_output = zeros(T, dim_pre, n_class, dim_post)
    for x = 1:dim_pre
      for y = 1:dim_post
        canonical_output[x, convert(Int,canonical_input[x,y])+1, y] = 1
      end
    end

    output = reshape(canonical_output, size(state.blobs[i]))
    got_output = to_array(state.blobs[i])

    @test all(-eps .< output - got_output .< eps)
  end

  shutdown(backend, state)
end

function test_index2onehot_layer(backend::Backend, n_input, T, eps)
  for td in [2,4,5]
    test_index2onehot_layer(backend, td, n_input, T, eps)
  end
end
function test_index2onehot_layer(backend::Backend)
  test_index2onehot_layer(backend, 3, Float32, 1e-5)
  test_index2onehot_layer(backend, 3, Float64, 1e-10)
end

if test_cpu
  test_index2onehot_layer(backend_cpu)
end
if test_gpu
  test_index2onehot_layer(backend_gpu)
end
