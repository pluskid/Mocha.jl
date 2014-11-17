function test_element_wise_layer(sys::System, op::ElementWiseFunctorType, jl_op::Function)
  println("-- Testing ElementWiseLayer{$op} on $(typeof(sys.backend))...")

  eps = 1e-5
  NArg = get_num_args(op)
  inputs = [rand(4,5,6,7) for i = 1:NArg]
  input_blobs = Blob[make_blob(sys.backend, x) for x in inputs]
  diff_blobs = Blob[make_blob(sys.backend, x) for x in inputs]

  layer = ElementWiseLayer(bottoms=Array(Symbol, NArg), tops=[:result], operation=op)
  state = setup(sys, layer, input_blobs, diff_blobs)

  forward(sys, state, input_blobs)
  expected_res = jl_op(inputs...)
  got_res = similar(inputs[1])
  copy!(got_res, state.blobs[1])
  @test all(abs(got_res - expected_res) .< eps)

  top_diff = rand(size(inputs[1]))
  copy!(state.blobs_diff[1], top_diff)

  backward(sys, state, input_blobs, diff_blobs)
  got_grads = map(diff_blobs) do blob
    arr = zeros(size(blob))
    copy!(arr, blob)
    arr
  end

  if jl_op == (+)
    for i = 1:length(got_grads)
      @test all(abs(got_grads[i] - top_diff) .< eps)
    end
  elseif jl_op == (-)
    @test all(abs(got_grads[1] - top_diff) .< eps)
    @test all(abs(got_grads[2] + top_diff) .< eps)
  elseif jl_op == (.*)
    @test all(abs(got_grads[1] - top_diff.*inputs[2]) .< eps)
    @test all(abs(got_grads[2] - top_diff.*inputs[1]) .< eps)
  elseif jl_op == (./)
    @test all(abs(got_grads[1] - top_diff./inputs[2]) .< eps)
    @test all(abs(got_grads[2] + top_diff.*inputs[1]./inputs[2]./inputs[2]) .< eps)
  else
    error("Unknown operation $jl_op")
  end
end

function test_element_wise_layer(sys::System)
  for (functor,jl_op) in ((ElementWiseFunctors.Add(), (+)),
                          (ElementWiseFunctors.Subtract(),(-)),
                          (ElementWiseFunctors.Multiply(), (.*)),
                          (ElementWiseFunctors.Divide(), (./)))
    test_element_wise_layer(sys, functor, jl_op)
  end
end

if test_cpu
  test_element_wise_layer(sys_cpu)
end
if test_cudnn
  test_element_wise_layer(sys_cudnn)
end

