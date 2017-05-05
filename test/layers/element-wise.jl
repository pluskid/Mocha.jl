function test_element_wise_layer(backend::Backend, op::ElementWiseFunctorType, jl_op::Function, T, eps)
  println("-- Testing ElementWiseLayer{$op} on $(typeof(backend)){$T}...")

  tensor_dim = abs(rand(Int)) % 6 + 1
  dims = tuple(rand(1:8, tensor_dim)...)
  println("    > $dims")

  NArg = get_num_args(op)
  inputs = [rand(T, dims) for i = 1:NArg]
  input_blobs = Blob[make_blob(backend, x) for x in inputs]
  diff_blobs = Blob[make_blob(backend, x) for x in inputs]

  layer = ElementWiseLayer(bottoms=Array{Symbol}(NArg), tops=[:result], operation=op)
  state = setup(backend, layer, input_blobs, diff_blobs)

  forward(backend, state, input_blobs)
  expected_res = jl_op.(inputs...)
  got_res = similar(inputs[1])
  copy!(got_res, state.blobs[1])
  @test all(abs.(got_res - expected_res) .< eps)

  top_diff = rand(T, size(inputs[1]))
  copy!(state.blobs_diff[1], top_diff)

  backward(backend, state, input_blobs, diff_blobs)
  got_grads = map(diff_blobs) do blob
    arr = zeros(T, size(blob))
    copy!(arr, blob)
    arr
  end

  if jl_op == (+)
    for i = 1:length(got_grads)
      @test all(abs.(got_grads[i] - top_diff) .< eps)
    end
  elseif jl_op == (-)
    @test all(abs.(got_grads[1] - top_diff) .< eps)
    @test all(abs.(got_grads[2] + top_diff) .< eps)
  elseif jl_op == (*)
    @test all(abs.(got_grads[1] - top_diff.*inputs[2]) .< eps)
    @test all(abs.(got_grads[2] - top_diff.*inputs[1]) .< eps)
  elseif jl_op == (/)
    @test all(abs.(got_grads[1] - top_diff./inputs[2]) .< eps)
    #@test all(abs.(got_grads[2] + top_diff.*inputs[1]./(inputs[2].*inputs[2])) .< eps)
    @test all(abs.(got_grads[2] + top_diff.*got_res./inputs[2]) .< eps)
  else
    error("Unknown operation $jl_op")
  end

  shutdown(backend, state)
end

function test_element_wise_layer(backend::Backend, T, eps)
  for (functor,jl_op) in ((ElementWiseFunctors.Add(), (+)),
                          (ElementWiseFunctors.Subtract(),(-)),
                          (ElementWiseFunctors.Multiply(), (*)),
                          (ElementWiseFunctors.Divide(), (/)))
    test_element_wise_layer(backend, functor, jl_op, T, eps)
  end
end

function test_element_wise_layer(backend::Backend)
  test_element_wise_layer(backend, Float32, 1e-1)
  test_element_wise_layer(backend, Float64, 1e-5)
end

if test_cpu
  test_element_wise_layer(backend_cpu)
end
if test_gpu
  test_element_wise_layer(backend_gpu)
end

