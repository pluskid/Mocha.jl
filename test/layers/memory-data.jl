function test_memory_data_layer(backend::Backend, T, eps)
  println("-- Testing Memory Data Layer on $(typeof(backend)){$T}...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size = 6
  tensor_dim = abs(rand(Int)) % 6 + 1
  data_dim = tuple(rand(1:6, tensor_dim)...)
  println("    > $data_dim")

  data = rand(T, data_dim..., 9)
  mean_data = rand(T, data_dim..., 1)
  mean_blob = make_blob(backend, mean_data)

  ############################################################
  # Setup
  ############################################################

  # batch size is determined by
  layer = MemoryDataLayer(data = Array[data], tops = [:data], batch_size=batch_size,
      transformers=[(:data, DataTransformers.SubMean(mean_blob=mean_blob))])
  state = setup(backend, layer, Blob[], Blob[])

  data_idx = map(x->1:x, data_dim)
  layer_data = Array{eltype(data)}(tuple(data_dim..., batch_size))

  data_aug = cat(tensor_dim+1, data, data)
  data_aug .-= mean_data
  forward(backend, state, Blob[])
  copy!(layer_data, state.blobs[1])
  @test all(-eps .< layer_data - data_aug[data_idx..., 1:1+batch_size-1] .< eps)
  @test state.epoch == 0

  forward(backend, state, Blob[])
  copy!(layer_data, state.blobs[1])
  @test all(-eps .< layer_data - data_aug[data_idx..., batch_size+1:2batch_size] .< eps)
  @test state.epoch == 1

  forward(backend, state, Blob[])
  copy!(layer_data, state.blobs[1])
  @test all(-eps .< layer_data - data_aug[data_idx..., 2batch_size+1:3batch_size] .< eps)
  @test state.epoch == 2

  shutdown(backend, state)
end
function test_memory_data_layer(backend::Backend)
  test_memory_data_layer(backend, Float32, 1e-5)
  test_memory_data_layer(backend, Float64, 1e-10)
end

if test_cpu
  test_memory_data_layer(backend_cpu)
end
if test_gpu
  test_memory_data_layer(backend_gpu)
end

