function test_rmna_layer(backend::Backend, T, NAval, eps)
  println("-- Testing RmNaLayer on $(typeof(backend)){$T}...")

  NAval = convert(T,NAval)

  tensor_dim = abs(rand(Int)) % 6 + 1
  dims = tuple((abs(rand(Int, tensor_dim)) % 8 + 1)...)
  println("    > $dims")

  ratio = convert(T, 0.6)
  input = rand(T, dims)
  diff = similar(input)

  for i in eachindex(input)
    ( rand() > ratio )  &&  ( input[i] = NAval )
  end

  if isnan(NAval)
    NanInd = isnan(input)
  else
    NanInd = input == NAval
  end

  nNan = sum(NanInd)

  input_blob = make_blob(backend, input)
  diff_blob  = make_blob(backend, diff)

  println("    > Setup")
  layer = RmNaLayer(NAval = NAval, bottoms=[:input])
  state = setup(backend, layer, Blob[input_blob], Blob[diff_blob])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(input))
  copy!(got_output, input_blob)

  expected_output = similar(input)
  for i in eachindex(input)
    expected_output[i] = NanInd[i] == one(T) ? zero(T) : input[i]
  end

  @test all(abs(got_output - expected_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(input))
  copy!(diff_blob, top_diff)
  backward(backend, state, Blob[input_blob], Blob[diff_blob])
  expected_grad = similar(top_diff)
  for i in eachindex(top_diff)
    expected_grad[i] = top_diff[i] * !NanInd[i]
  end
  got_grad = zeros(T, size(expected_grad))
  copy!(got_grad, diff_blob)

  @show got_grad
  @show expected_grad
  @test all(abs(got_grad - expected_grad) .< eps)

  shutdown(backend, state)
end
function test_rmna_layer(backend::Backend)
  test_rmna_layer(backend, Float64, NaN64, 1e-10)
  test_rmna_layer(backend, Float32, NaN32, 1e-4)
  test_rmna_layer(backend, Float64, NaN64, 1e-10)
  test_rmna_layer(backend, Float32, NaN32, 1e-4)
end

if test_cpu
  test_rmna_layer(backend_cpu)
end
if test_gpu
  test_rmna_layer(backend_gpu)
end
