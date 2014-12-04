using HDF5

function test_hdf5_data_layer(backend::Backend, T, eps)
  println("-- Testing HDF5 Data Layer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size = 3
  data_dim = (7,8,2)

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
  state = setup(backend, layer, Blob[], Blob[])
  @test state.epoch == 0

  data = cat(4, data_all...)
  data = cat(4, data, data, data)
  data = data * scale

  data_idx = map(x->1:x, data_dim)
  layer_data = Array(eltype(data), tuple(data_dim..., batch_size))
  for i = 1:batch_size:size(data,4)-batch_size+1
    forward(backend, state, Blob[])
    copy!(layer_data, state.blobs[1])
    @test all(-eps .< layer_data - data[data_idx..., i:i+batch_size-1] .< eps)
  end
  @test state.epoch == 3

  ############################################################
  # Clean up
  ############################################################
  shutdown(backend, state)
  rm(source_fn)
  for fn in h5fn_all
    rm(fn)
  end
end

function test_hdf5_data_layer_shuffle(backend::Backend, batch_size, n, T)
  println("-- Testing HDF5 Data Layer (shuffle,n=$n,b=$batch_size) on $(typeof(backend)){$T}...")

  # To test random shuffling, we generate a dataset containing integer 1:n.
  # Then we run HDF5 layer n times forward, and collect all the output data.
  # For each integer i in 1:n, it should show up exactly b times, where b
  # is the batch size.

  data = reshape(convert(Array{T}, collect(1:n)), 1, 1, 1, n)
  h5fn = string(tempname(), ".hdf5")
  h5open(h5fn, "w") do file
    file["data"] = data
  end

  source_fn = string(tempname(), ".txt")
  open(source_fn, "w") do file
    println(file, h5fn)
  end

  layer = HDF5DataLayer(source=source_fn, tops=[:data], batch_size=batch_size, shuffle=true)
  state = setup(backend, layer, Blob[], Blob[])

  data_got_all = Int[]
  data_got = zeros(T, 1, 1, 1, batch_size)
  for i = 1:n
    forward(backend, state, Blob[])
    copy!(data_got, state.blobs[1])
    append!(data_got_all, convert(Vector{Int}, data_got[:]))
  end

  for i = 1:n
    @test sum(data_got_all .== i) == batch_size
  end

  if data_got_all[1:n] == collect(1:n)
    println("WARNING: data not shuffled, is today a lucky day or is there a bug?")
  end

  shutdown(backend, state)
  rm(source_fn)
  rm(h5fn)
end

function test_hdf5_data_layer_shuffle(backend::Backend, T)
  test_hdf5_data_layer_shuffle(backend, 2, 3, T)
  test_hdf5_data_layer_shuffle(backend, 3, 2, T)
end

function test_hdf5_data_layer(backend::Backend)
  test_hdf5_data_layer(backend, Float32, 1e-5)
  test_hdf5_data_layer(backend, Float64, 1e-10)
  test_hdf5_data_layer_shuffle(backend, Float32)
  test_hdf5_data_layer_shuffle(backend, Float64)
end

if test_cpu
  test_hdf5_data_layer(backend_cpu)
end
if test_gpu
  test_hdf5_data_layer(backend_gpu)
end
