using HDF5

function test_hdf5_data_layer(sys::System, T, eps)
  println("-- Testing HDF5 Data Layer on $(typeof(sys.backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size = 3
  data_dim = (1,1,2)

  data_all = [rand(T, data_dim..., x) for x in [5 1 2]]
  h5fn_all = [string(tempname(), ".hdf5") for x in 1:length(data_all)]

  for i = 1:length(data_all)
    h5 = h5open(h5fn_all[i], "w")
    h5["data"] = data_all[i]
    close(h5)
  end

  source_fn = string(tempname(), ".txt")
  open(source_fn, "w") do s
    for fn in h5fn_all
      println(s, fn)
    end
  end

  ############################################################
  # Setup
  ############################################################

  scale = rand()
  layer = HDF5DataLayer(source = source_fn, tops = [:data], batch_size=batch_size,
      transformers=[(:data, DataTransformers.Scale(scale))])
  state = setup(sys, layer, Blob[], Blob[])
  @test state.epoch == 0

  data = cat(4, data_all...)
  data = cat(4, data, data, data)
  data = data * scale

  data_idx = map(x->1:x, data_dim)
  layer_data = Array(eltype(data), tuple(data_dim..., batch_size))
  for i = 1:batch_size:size(data,4)-batch_size+1
    forward(sys, state, Blob[])
    copy!(layer_data, state.blobs[1])
    @test all(-eps .< layer_data - data[data_idx..., i:i+batch_size-1] .< eps)
  end
  @test state.epoch == 3

  ############################################################
  # Clean up
  ############################################################
  shutdown(sys, state)
  rm(source_fn)
  for fn in h5fn_all
    rm(fn)
  end
end

function test_hdf5_data_layer(sys::System)
  test_hdf5_data_layer(sys, Float32, 1e-5)
  test_hdf5_data_layer(sys, Float64, 1e-10)
end

if test_cpu
  test_hdf5_data_layer(sys_cpu)
end
if test_cudnn
  test_hdf5_data_layer(sys_cudnn)
end
