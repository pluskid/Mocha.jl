using HDF5

function test_hdf5_output_layer(sys::System, T, eps)
  println("-- Testing HDF5 Output Layer on $(typeof(sys.backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  width, height, channels, batch_size = (5,6,7,8)
  input = rand(T, width, height, channels, batch_size)
  input_blob = make_blob(sys.backend, input)

  output_fn = string(tempname(), ".hdf5")
  layer = HDF5OutputLayer(bottoms=[:input], datasets=[:foobar], filename=output_fn)
  state = setup(sys, layer, Blob[input_blob], Blob[NullBlob()])

  # repeat 3 times
  forward(sys, state, Blob[input_blob])
  forward(sys, state, Blob[input_blob])
  forward(sys, state, Blob[input_blob])

  shutdown(sys, state)

  expected_output = cat(4, input, input, input)
  got_output = h5open(output_fn, "r") do h5
    read(h5, "foobar")
  end

  @test size(expected_output) == size(got_output)
  @test eltype(expected_output) == eltype(got_output)
  @test all(abs(expected_output-got_output) .< eps)

  rm(output_fn)
end

function test_hdf5_output_layer(sys::System)
  test_hdf5_output_layer(sys, Float32, 1e-5)
  test_hdf5_output_layer(sys, Float64, 1e-10)
end

if test_cpu
  test_hdf5_output_layer(sys_cpu)
end
if test_cudnn
  test_hdf5_output_layer(sys_cudnn)
end

