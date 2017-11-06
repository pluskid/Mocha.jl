using HDF5

function test_hdf5_output_layer(backend::Backend, T, eps)
  println("-- Testing HDF5 Output Layer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  tensor_dim = abs(rand(Int)) % 4 + 2
  dims = tuple(rand(1:8, tensor_dim)...)
  println("    > $dims")

  input = rand(T, dims)
  input_blob = make_blob(backend, input)

  output_fn = string(Mocha.temp_filename(), ".hdf5")
  open(output_fn, "w") do file
    # create an empty file
  end
  layer = HDF5OutputLayer(bottoms=[:input], datasets=[:foobar], filename=output_fn)
  @test_throws ErrorException setup(backend, layer, Blob[input_blob], Blob[NullBlob()])

  layer = HDF5OutputLayer(bottoms=[:input], datasets=[:foobar],
      filename=output_fn, force_overwrite=true)
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])

  # repeat 3 times
  forward(backend, state, Blob[input_blob])
  forward(backend, state, Blob[input_blob])
  forward(backend, state, Blob[input_blob])

  shutdown(backend, state)

  expected_output = cat(tensor_dim, input, input, input)
  got_output = h5open(output_fn, "r") do h5
    read(h5, "foobar")
  end

  @test size(expected_output) == size(got_output)
  @test eltype(expected_output) == eltype(got_output)
  @test all(abs.(expected_output-got_output) .< eps)

  rm(output_fn)
end

function test_hdf5_output_layer(backend::Backend)
  test_hdf5_output_layer(backend, Float32, 1e-5)
  test_hdf5_output_layer(backend, Float64, 1e-10)
end

if test_cpu
  test_hdf5_output_layer(backend_cpu)
end
if test_gpu
  test_hdf5_output_layer(backend_gpu)
end

