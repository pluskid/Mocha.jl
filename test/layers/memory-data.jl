function test_memory_data_layer(sys::System)
  println("-- Testing Memory Data Layer on $(typeof(sys.backend))...")

  ############################################################
  # Prepare Data for Testing
  ############################################################
  batch_size = 3
  data_dim = (2,3,4)
  eps = 1e-15

  data = rand(data_dim..., 100)

  ############################################################
  # Setup
  ############################################################

  # batch size is determined by
  layer = MemoryDataLayer(data = Array[data], tops = [:data], batch_size=batch_size)
  state = setup(sys, layer, Blob[])

  data_idx = map(x->1:x, data_dim)
  layer_data = Array(eltype(data), tuple(data_dim..., batch_size))
  for i = 1:batch_size:size(data,1)-batch_size
    forward(sys, state, Blob[])
    copy!(layer_data, state.blobs[1])
    @test all(-eps .< layer_data - data[data_idx..., i:i+batch_size-1] .< eps)
  end
end

if test_cpu
  test_memory_data_layer(sys_cpu)
end
if test_cudnn
  test_memory_data_layer(sys_cudnn)
end

