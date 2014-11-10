using HDF5

function test_hdf5_data_layer(sys::System)
  println("-- Testing HDF5 Data Layer on $(typeof(sys.backend))...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size = 3
  data_dim = (2,3,4)
  eps = 1e-15

  data_all = [rand(data_dim..., x) for x in [5 1 2]]
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

  # batch size is determined by
  layer = HDF5DataLayer(; source = source_fn, tops = [:data], batch_size=batch_size)
  state = setup(sys, layer, Blob[])

  data = cat(4, data_all...)
  data = cat(4, data, data)

  data_idx = map(x->1:x, data_dim)
  layer_data = Array(eltype(data), tuple(data_dim..., batch_size))
  for i = 1:batch_size:size(data,1)-batch_size
    forward(sys, state, Blob[])
    copy!(layer_data, state.blobs[1])
    @test all(-eps .< layer_data - data[data_idx..., i:i+batch_size-1] .< eps)
  end
end

if test_cudnn
  test_hdf5_data_layer(sys_cudnn)
end
if test_cpu
  test_hdf5_data_layer(sys_cpu)
end
