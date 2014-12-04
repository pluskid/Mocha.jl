using HDF5

function test_hdf5_output_layer(backend::Backend, T, eps)
  println("-- Testing HDF5 Output Layer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  width, height, channels, batch_size = (5,6,7,8)
  input = rand(T, width, height, channels, batch_size)
  input_blob = make_blob(backend, input)

  output_fn = string(tempname(), ".hdf5")
  layer = HDF5OutputLayer(bottoms=[:input], datasets=[:foobar], filename=output_fn)
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])

  # repeat 3 times
  forward(backend, state, Blob[input_blob])
  forward(backend, state, Blob[input_blob])
  forward(backend, state, Blob[input_blob])

  shutdown(backend, state)

  expected_output = cat(4, input, input, input)
  got_output = h5open(output_fn, "r") do h5
    read(h5, "foobar")
  end

  @test size(expected_output) == size(got_output)
  @test eltype(expected_output) == eltype(got_output)
  @test all(abs(expected_output-got_output) .< eps)

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

