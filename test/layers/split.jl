function test_split_layer(backend::Backend, T)
  println("-- Testing SplitLayer on $(typeof(backend)){$T}...")

  eps = 1e-10
  input = rand(T, 3, 4, 5, 6)
  input_blob = make_blob(backend, input)

  println("    > Setup")
  layer = SplitLayer(tops=[:sp1, :sp2], bottoms=[:input])
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])
  @test length(state.blobs_diff) == 2
  @test all(map(x -> isa(x, NullBlob), state.blobs_diff))

  diff_blob = make_blob(backend, T, size(input))
  state = setup(backend, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(input))
  for i = 1:2
    copy!(got_output, state.blobs[i])
    @test all(abs(got_output - input) .< eps)
  end

  println("    > Backward")
  top_diff = Array{T}[rand(T, size(input)), rand(T, size(input))]
  for i = 1:2
    copy!(state.blobs_diff[i], top_diff[i])
  end
  backward(backend, state, Blob[input_blob], Blob[diff_blob])
  expected_grad = top_diff[1] + top_diff[2]
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, diff_blob)
  @test all(abs(got_grad - expected_grad) .< eps)

  shutdown(backend, state)
end
function test_split_layer(backend::Backend)
  test_split_layer(backend, Float32)
  test_split_layer(backend, Float64)
end

if test_cpu
  test_split_layer(backend_cpu)
end
if test_cudnn
  test_split_layer(backend_gpu)
end


