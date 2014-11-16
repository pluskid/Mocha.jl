function test_channel_pooling_layer(sys::System, pooling::PoolingFunction)
  println("-- Testing Pooling($(typeof(pooling))) on $(typeof(sys.backend))...")
  println("    > Setup")

  width, height, channels, num = (5, 6, 7, 8)
  pad = (1,2)
  kernel = 3
  stride = 2
  eps = 1e-10

  layer = ChannelPoolingLayer(kernel=kernel, stride=stride, pad=pad,
      tops=[:top], bottoms=[:bottom], pooling=pooling)

  input_dim = (width, height, channels, num)
  input = rand(input_dim)
  inputs = Blob[make_blob(sys.backend, input)]
  diffs = Blob[make_blob(sys.backend, input)]

  state = setup(sys, layer, inputs, diffs)

  println("    > Forward")
  forward(sys, state, inputs)
end

function test_channel_pooling_layer(sys::System)
  test_channel_pooling_layer(sys, Pooling.Max())
  test_channel_pooling_layer(sys, Pooling.Mean())
end

if test_cpu
  test_channel_pooling_layer(sys_cpu)
end
if test_cudnn
  test_channel_pooling_layer(sys_cudnn)
end
