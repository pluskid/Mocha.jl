function test_split_layer(sys::System)
  println("-- Testing SplitLayer on $(typeof(sys.backend))...")

  eps = 1e-10
  input = rand(3, 4, 5, 6)
  input_blob = make_blob(sys.backend, input)

  println("    > Setup")
  layer = SplitLayer(tops=[:sp1, :sp2], bottoms=[:input])
  state = setup(sys, layer, Blob[input_blob], Blob[NullBlob()])
  @test length(state.blobs_diff) == 2
  @test all(map(x -> isa(x, NullBlob), state.blobs_diff))

  diff_blob = make_blob(sys.backend, Float64, size(input))
  state = setup(sys, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(sys, state, Blob[input_blob])
  got_output = zeros(size(input))
  for i = 1:2
    copy!(got_output, state.blobs[i])
    @test all(abs(got_output - input) .< eps)
  end

  println("    > Backward")
  top_diff = Array{Float64}[rand(size(input)), rand(size(input))]
  for i = 1:2
    copy!(state.blobs_diff[i], top_diff[i])
  end
  backward(sys, state, Blob[input_blob], Blob[diff_blob])
  expected_grad = top_diff[1] + top_diff[2]
  got_grad = zeros(size(expected_grad))
  copy!(got_grad, diff_blob)
  @test all(abs(got_grad - expected_grad) .< eps)

  shutdown(sys, state)
end

if test_cpu
  test_split_layer(sys_cpu)
end
if test_cudnn
  test_split_layer(sys_cudnn)
end


