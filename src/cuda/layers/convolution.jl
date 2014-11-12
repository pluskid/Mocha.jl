type CuDNNConvState
  inputs_desc   :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  conv_desc     :: Vector{CuDNN.ConvolutionDescriptor}
  filter_desc   :: CuDNN.FilterDescriptor
  bias_desc     :: CuDNN.Tensor4dDescriptor

  bottom_offset :: Int
  top_offset    :: Int
  weight_offset :: Int
  bias_offset   :: Int
end

function setup_etc(sys::System{CuDNNBackend}, layer::ConvolutionLayer, dtype, width, height, channels, width_out, height_out)
  filter_desc = CuDNN.create_filter_descriptor(dtype, (layer.kernel[1], layer.kernel[2],
      div(channels,layer.n_group), div(layer.n_filter,layer.n_group)))
  bias_desc = CuDNN.create_tensor4d_descriptor(dtype, (1,1,div(layer.n_filter,layer.n_group),1))

  inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  conv_desc = Array(CuDNN.ConvolutionDescriptor, length(inputs))
  for i = 1:length(inputs)
    inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width,height,div(channels,layer.n_group),batch_size),
        (1, width, width*height, width*height*channels))
    outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width_out, height_out, div(layer.n_filter,layer.n_group), batch_size),
        (1, width_out, width_out*height_out, width_out*height_out*layer.n_filter))
    conv_desc[i] = CuDNN.create_convolution_descriptor(inputs_desc[i], filter_desc, layer.pad,
        layer.stride, (1,1), CuDNN.CUDNN_CROSS_CORRELATION)
  end

  bottom_offset = div(channels,layer.n_group) * height * width * sizeof(dtype)
  top_offset = div(layer.n_filter,layer.n_group) * height_out * width_out * sizeof(dtype)
  weight_offset = div(layer.n_filter,layer.n_group) * div(channels,layer.n_group) * layer.kernel[1] * layer.kernel[2] * sizeof(dtype)
  bias_offset = div(layer.n_filter,layer.n_group) * sizeof(dtype)

  etc = CuDNNConvState(inputs_desc, outputs_desc, conv_desc, filter_desc, bias_desc,
      bottom_offset, top_offset, weight_offset, bias_offset)
  return etc
end

function forward(sys::System{CuDNNBackend}, state::ConvolutionLayerState, inputs::Vector{Blob})
  one = convert(eltype(inputs[1]), 1)

  for i = 1:length(inputs)
    for g = 1:state.layer.n_group
      input_ptr = CuPtr(inputs[i].ptr.p + state.etc.bottom_offset * (g-1))
      output_ptr = CuPtr(state.blobs[i].ptr.p + state.etc.top_offset * (g-1))
      filter_ptr = CuPtr(state.filter.ptr.p + state.etc.weight_offset * (g-1))

      CuDNN.convolution_forward(sys.backend.cudnn_ctx, state.etc.inputs_desc[i], input_ptr,
          state.etc.filter_desc, filter_ptr, state.etc.conv_desc[i],
          state.etc.outputs_desc[i], output_ptr, CuDNN.CUDNN_RESULT_NO_ACCUMULATE)

      # bias
      CuDNN.add_tensor4d(sys.backend.cudnn_ctx, CuDNN.CUDNN_ADD_SAME_C, one,
          state.etc.bias_desc, CuPtr(state.bias.ptr.p + state.etc.bias_offset * (g-1)),
          state.etc.outputs_desc[i], output_ptr)
    end
  end
end

function backward(sys::System{CuDNNBackend}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  erase!(state.∇filter)
  erase!(state.∇bias)

  for i = 1:length(inputs)
    bottom = inputs[i]
    top_diff = state.blobs_diff[i]

    for g = 1:state.layer.n_group
      # gradient w.r.t. bias
      CuDNN.convolution_backward_bias(sys.backend.cudnn_ctx,
          state.etc.outputs_desc[i], CuPtr(top_diff.ptr.p + state.etc.top_offset * (g-1)),
          state.etc.bias_desc, CuPtr(state.∇bias.ptr.p + state.etc.bias_offset * (g-1)),
          CuDNN.CUDNN_RESULT_ACCUMULATE)

      # gradient w.r.t. weights
      CuDNN.convolution_backward_filter(sys.backend.cudnn_ctx,
          state.etc.inputs_desc[i], CuPtr(bottom.ptr.p + state.etc.bottom_offset * (g-1)),
          state.etc.outputs_desc[i], CuPtr(top_diff.ptr.p + state.etc.top_offset * (g-1)),
          state.etc.conv_desc[i],
          state.etc.filter_desc, CuPtr(state.∇filter.ptr.p + state.etc.weight_offset * (g-1)),
          CuDNN.CUDNN_RESULT_ACCUMULATE)

      # gradient w.r.t. bottom data
      if isa(diffs[i], CuTensorBlob)
        CuDNN.convolution_backward_data(sys.backend.cudnn_ctx,
            state.etc.filter_desc, CuPtr(state.filter.ptr.p + state.etc.weight_offset * (g-1)),
            state.etc.outputs_desc[i], CuPtr(top_diff.ptr.p + state.etc.top_offset * (g-1)),
            state.etc.conv_desc[i],
            state.etc.inputs_desc[i], CuPtr(diffs[i].ptr.p + state.etc.bottom_offset * (g-1)),
            CuDNN.CUDNN_RESULT_NO_ACCUMULATE)
      end
    end
  end
end
