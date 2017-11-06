function test_identity_layer(backend::Backend, T)
  println("-- Testing IdentityLayer on $(typeof(backend)){$T}...")

  eps = 1e-10
  input = rand(T, 3, 4, 5)
  input_blob = make_blob(backend, input)

  println("    > Setup")
  layer = IdentityLayer(tops=[:foo], bottoms=[:bar])
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])
  @test all(map(x -> isa(x, NullBlob), state.blobs_diff))

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = to_array(state.blobs[1])
  @test all(abs.(got_output-input) .< eps)

  shutdown(backend, state)
end

function test_identity_layer(backend::Backend)
  test_identity_layer(backend, Float32)
  test_identity_layer(backend, Float64)
end

if test_cpu
  test_identity_layer(backend_cpu)
end
if test_gpu
  test_identity_layer(backend_gpu)
end

