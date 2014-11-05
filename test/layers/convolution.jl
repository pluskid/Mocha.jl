function test_convolution_layer(sys::System)
  println("-- Testing Convolution on $(typeof(sys.backend))...")
  println("    > Setup")
  input_w = 16
  input_h = 10
  input_chann = 6
  input_num = 24
  n_group = 2
  filter_w = 3
  filter_h = 4
  n_filter = 12
  eps = 1e-10

  input_dims = (input_w, input_h, input_chann, input_num)
  filter_dims = (filter_w, filter_h, int(input_chann/n_group), n_filter)
  bias_dims = (1, 1, n_filter, 1)

  layer = ConvolutionLayer(; kernel=(filter_w, filter_h), stride=(1,2), pad=(2,2), n_filter=n_filter, n_group=n_group,
      tops=String["conv"], bottoms=String["data"])

  input = rand(input_dims)
  inputs = Array(Blob, 1)
  if isa(sys.backend, CPUBackend)
    error("TODO")
  elseif isa(sys.backend, CuDNNBackend)
    inputs = Blob[Mocha.cudnn_make_tensor_blob(Float64, input_dims)]
    copy!(inputs[1], input)
  end

  state = setup(sys, layer, inputs)

  # test that we are getting the correct output shape
  out_blob_dims = CuDNN.get_output_tensor4d_dim(state.etc.conv_desc[1], CuDNN.CUDNN_CONVOLUTION_FWD)
  @test out_blob_dims == (get_width(state.blobs[1]), get_height(state.blobs[1]), int(get_chann(state.blobs[1])/n_group), input_num)

  println("    > Forward")
  filter = rand(filter_dims)
  copy!(state.filter, filter)
  bias = rand(bias_dims)
  copy!(state.bias, bias)

  forward(sys, state, inputs)
  expected_output = convolution_forward(state, filter, bias, input)
  @test size(expected_output) == size(state.blobs[1])
  got_output = similar(expected_output)
  copy!(got_output, state.blobs[1])
  @test all(-eps .< expected_output - got_output .< eps)
end

# naive implementation of convolution forward, used to check the correctness
function convolution_forward(state, filter::Array, bias::Array, input::Array)
  width, height, channel, num = size(input)
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  output = Array(eltype(input), size(state.blobs[1]))
  output[:] = 0

  for n = 1:num
    for g = 1:n_group
      for o = 1:o_g
        for k = 1:k_g
          for y = 1:state.height_out
            for x = 1:state.width_out
              for p = 1:state.layer.kernel[2]
                for q = 1:state.layer.kernel[1]
                  in_y = (y-1) * state.layer.stride[2] - state.layer.pad[2] + p
                  in_x = (x-1) * state.layer.stride[1] - state.layer.pad[1] + q
                  if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
                    output[x, y, (g-1)*o_g+o, n] += input[in_x, in_y, (g-1)*k_g+k, n] *
                        filter[q, p, k, (g-1)*o_g+o]
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  for n = 1:num
    for o = 1:state.layer.n_filter
      for y = 1:state.height_out
        for x = 1:state.width_out
          output[x, y, o, n] += bias[o]
        end
      end
    end
  end

  return output
end

if test_cpu
  test_convolution_layer(sys_cpu)
end
if test_cudnn
  test_convolution_layer(sys_cudnn)
end
