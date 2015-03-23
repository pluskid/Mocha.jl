function test_net_topology_duplicated_blob(backend::Backend)
  println("-- Testing network topology with duplicated blobs")
  layer = MemoryDataLayer(tops=[:data, :data], batch_size=1, data=Array[rand(1),rand(1)])
  @test_throws TopologyError Net("net", backend, Layer[layer])

  layer1 = MemoryDataLayer(tops=[:data], batch_size=1, data=Array[rand(1,1,1,1)])
  layer2 = ReshapeLayer(tops=[:data], bottoms=[:data], shape=(1,1,1))
  @test_throws TopologyError Net("net", backend, Layer[layer1, layer2])
end
function test_net_topology_missing_blob(backend::Backend)
  println("-- Testing network topology with missing blobs")
  layer = ReshapeLayer(tops=[:output], bottoms=[:input], shape=(1,1,1))
  @test_throws TopologyError Net("net", backend, Layer[layer])
end
function test_net_topology_loop(backend::Backend)
  println("-- Testing network topology with circular dependency")
  layer1 = InnerProductLayer(name="ip1", tops=[:output], bottoms=[:input], output_dim=1)
  layer2 = InnerProductLayer(name="ip2", tops=[:input], bottoms=[:output], output_dim=1)
  @test_throws TopologyError Net("net", backend, Layer[layer1, layer2])
end
function test_net_topology_multiple_bp(backend::Backend)
  println("-- Testing network topology with multiple back-propagate path")

  #----
  # This one is OK, because although the data blob is shared by two inner-product
  # layer, since the data blob do not need back-propagation, there won't be a
  # problem
  println("    > Good blob sharing")
  layer1 = MemoryDataLayer(tops=[:data], batch_size=1, data=Array[rand(1,1,1,1)])
  layer_rs = ReshapeLayer(tops=[:data_rs], bottoms=[:data], shape=(1,1,1))
  layer_ip1 = InnerProductLayer(tops=[:ip1], bottoms=[:data_rs], output_dim=1, name="ip1")
  layer_ip2 = InnerProductLayer(tops=[:ip2], bottoms=[:data_rs], output_dim=1, name="ip2")
  layer_loss = SquareLossLayer(bottoms=[:ip1,:ip2])
  registry_reset(backend)
  net = Net("net", backend, Layer[layer1, layer_rs, layer_ip1, layer_ip2, layer_loss])
  @test check_bp_topology(net)
  destroy(net)

  #----
  # This one is bad because the output of ip0 is being shared by ip1 and ip2, when
  # both ip1 and ip2 try to back-propagate into ip0, there will be a conflict. A
  # SplitLayer is explicitly needed here.
  println("    > Bad blob sharing")
  layer0 = MemoryDataLayer(tops=[:raw_data], batch_size=1, data=Array[rand(1,1,1,1)])
  layer_ip0 = InnerProductLayer(tops=[:data], bottoms=[:raw_data], output_dim=1, name="ip0")
  registry_reset(backend)
  net = Net("net", backend, Layer[layer0, layer_ip0, layer_rs, layer_ip1, layer_ip2, layer_loss])
  @test_throws TopologyError check_bp_topology(net)
end
function test_net_topology_dangling_blob(backend::Backend)
  tmp_h5 = Mocha.temp_filename()
  #----
  # This one is good, the ip1 layer is connected with a loss layer.
  println("-- Testing network topology with dangling blob")
  println("    > Good case")
  layer1 = MemoryDataLayer(tops=[:data,:label], batch_size=1, data=Array[rand(1,1,1,1),rand(1,1,1,1)])
  layer_ip1 = InnerProductLayer(tops=[:ip1], bottoms=[:data], output_dim=1, name="ip1")
  layer_out = HDF5OutputLayer(bottoms=[:data], filename=tmp_h5)
  layer_loss = SquareLossLayer(bottoms=[:ip1,:label])
  registry_reset(backend)
  net = Net("net", backend, [layer1, layer_ip1, layer_out, layer_loss])
  @test check_bp_topology(net)
  destroy(net)
  if isfile(tmp_h5)
    rm(tmp_h5)
  end

  #----
  # This one is bad, because the ip2 blob requires bp, but not connected in a valid bp path
  println("    > Bad case")
  layer_ip2 = InnerProductLayer(tops=[:ip2], bottoms=[:data], output_dim=1, name="ip2")
  registry_reset(backend)
  net = Net("net", backend, [layer1, layer_ip1, layer_ip2, layer_loss])
  @test_throws TopologyError check_bp_topology(net)
  destroy(net)

  #--
  # This one is good, although the data2 blob is dangling, but it actually comes directly
  # from a data layer, thus no bp is needed. In this case, the SplitLayer could actually
  # be omitted
  println("    > Good case 2")
  layer_split = SplitLayer(bottoms=[:data], tops=[:data1, :data2])
  layer_ip3 = InnerProductLayer(tops=[:ip3], bottoms=[:data1], output_dim=1, name="ip3")
  layer_out2 = HDF5OutputLayer(bottoms=[:data2], filename=tmp_h5)
  layer_loss2 = SquareLossLayer(bottoms=[:ip3,:label])
  registry_reset(backend)
  net = Net("net", backend, [layer1, layer_split, layer_ip3, layer_out2, layer_loss2])
  @test check_bp_topology(net)
  destroy(net)
  if isfile(tmp_h5)
    rm(tmp_h5)
  end

  #--
  # This one is bad, in a split layer, both of the output blobs data1 and data2 require
  # bp, but data2 is only connected to an HDF5 output layer, which does not do bp
  println("    > Bad case 2")
  layer0 = MemoryDataLayer(tops=[:data0,:label], batch_size=1,data=Array[rand(1,1,1,1),rand(1,1,1,1)])
  layer_ip0 = InnerProductLayer(tops=[:data], bottoms=[:data0], output_dim=1, name="ip0")
  registry_reset(backend)
  net = Net("net", backend, [layer0, layer_ip0, layer_split, layer_ip3, layer_out2, layer_loss2])
  @test_throws TopologyError check_bp_topology(net)
  destroy(net)
  if isfile(tmp_h5)
    rm(tmp_h5)
  end
end

function test_net_topology(backend::Backend)
  test_net_topology_duplicated_blob(backend)
  test_net_topology_missing_blob(backend)
  test_net_topology_loop(backend)
  test_net_topology_multiple_bp(backend)
  test_net_topology_dangling_blob(backend)
end

if test_cpu
  test_net_topology(backend_cpu)
end
if test_gpu
  test_net_topology(backend_gpu)
end
