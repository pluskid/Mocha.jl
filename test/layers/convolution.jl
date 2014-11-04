function test_convolution_layer(sys::System)
  println("-- Testing InnerProductLayer...")
  println("    > Setup")
  input_w = 16
  input_h = 10
  input_chann = 6
  input_num = 24
  n_group = 2
  filter_w = 3
  filter_h = 4

  input_dims = (input_w, input_h, input_chann, input_num)
  filter_dims = (filter_w, filter_h, input_chann, 1)
  bias_dims = (1, 1, input_chann, 1)

  layer = ConvolutionLayer(; kernel=(filter_w, filter_h), stride=(1,2), pad=(2,2), n_filter=12, n_group=n_group, 
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
end

# naive implementation of convolution forward, used to check the correctness
function convolution_forward(state, filter::Array, bias::Array, input::Array)
  width, height, channel, num = size(input)
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  output = Array(eltype(input), size(state.blobs[1]))

  for n = 1:num
    for g = 1:n_group
      for o = 1:o_g
        for k = 1:k_g
          for y = 1:state.height_out
            for x = 1:state.width_out
              output[x, y, (g-1)*o_g+o, n] = bias[(g-1)*o_g+o]

              for p = 1:state.layer.kernel[1]
                for q = 1:state.layer.kernel[2]
                  in_y = (y-1) * state.layer.stride[1] - state.layer.pad[1] + p
                  in_x = (x-1) * state.layer.stride[2] - state.layer.pad[2] + q
                  if (in_y >= 1 && in_y <= height && in_x >= 1 && in_y <= width)
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

  return output
end

