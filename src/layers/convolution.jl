@defstruct ConvolutionLayer CompLayer (
  (kernel :: NTuple{2,Int} = (1,1), length(kernel)==2 && all([kernel...] .> 0)),
  (stride :: NTuple{2,Int} = (1,1), length(stride)==2 && all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), length(pad)==2 && all([pad...] .>= 0)),
  (n_filter :: Int = 1, n_filter > 0),
  (n_group :: Int = 1, n_group > 0),
  neuron :: ActivationFunction = Neurons.Identity(),
  (bottoms :: Vector{String} = String[], length(bottoms) > 0),
  (tops :: Vector{String} = String[], length(tops) == length(bottoms)),
  filter_init :: Initializer = ConstantInitializer(0),
  bias_init :: Initializer = ConstantInitializer(0),
  filter_regu :: Regularizer = NoRegu(),
  bias_regu :: Regularizer = NoRegu()
)

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

type ConvolutionLayerState <: LayerState
  layer      :: ConvolutionLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  filter  :: Blob
  ∇filter :: Blob
  bias    :: Blob
  ∇bias   :: Blob

  ConvolutionLayerState(sys::System, layer::ConvolutionLayer, inputs::Vector{Blob}) = begin
    channels = get_chann(inputs[1])
    @assert channels % layer.n_group == 0
    @assert layer.n_filter % layer.n_group == 0

    batch_size = get_num(inputs[1])
    height = get_height(inputs[1])
    width  = get_width(inputs[1])
    width_out  = int((width  + 2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]) + 1
    height_out = int((height + 2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]) + 1

    dtype = eltype(inputs[1])

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))
    if isa(sys.backend, CPUBackend)
      for i = 1:length(inputs)
        blobs[i] = CPUBlob(Array(dtype, size(inputs[i],1), layer.n_filter, height_out, width_out))
        blobs_diff[i] = CPUBlob(similar(blobs[i].data))
      end

      filter = CPUBlob(Array(dtype, layer.n_filter, int(channels/layer.n_group), layer.kernel...))
      ∇filter = CPUBlob(similar(filter.data))
      bias = CPUBlob(Array(dtype, layer.n_filter))
      ∇bias = CPUBlob(similar(bias.data))

      col_buffer = CPUBlob(Array(dtype, 1, channels*prod(layer.kernel), height_out, width_out))
    elseif isa(sys.backend, CuDNNBackend)
      for i = 1:length(inputs)
        blobs[i] = cudnn_make_tensor_blob(dtype, width_out, height_out, layer.n_filter, batch_size)
        blobs_diff[i] = cudnn_make_tensor_blob(dtype, width_out, height_out, layer.n_filter, batch_size)
      end

      filter = cudnn_make_tensor_blob(dtype, layer.kernel[1], layer.kernel[2], 
          int(channels/layer.n_group), layer.n_filter)
      ∇filter = cudnn_make_tensor_blob(dtype, layer.kernel[1], layer.kernel[2], 
          int(channels/layer.n_group), layer.n_filter)
      bias = cudnn_make_tensor_blob(dtype, layer.n_filter)
      ∇bias = cudnn_make_tensor_blob(dtype, layer.n_filter)

      filter_desc = CuDNN.create_filter_descriptor(dtype, (layer.kernel[1], layer.kernel[2],
          int(channels/layer.n_group), int(layer.n_filter/layer.n_group)))
      bias_desc = CuDNN.create_tensor4d_descriptor(dtype, (1,1,int(layer.n_filter/layer.n_group),1))

      inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      conv_desc = Array(CuDNN.ConvolutionDescriptor, length(inputs))
      for i = 1:length(inputs)
        inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width,height,int(channels/layer.n_group),batch_size), 
            (1, width, width*height, width*height*channels))
        outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width_out, height_out, int(layer.n_filter/layer.n_group), batch_size),
            (1, width_out, width_out*height_out, width_out*height_out*layer.n_filter))
        conv_desc[i] = CuDNN.create_convolution_descriptor(inputs_desc[i], filter_desc, layer.pad,
            layer.stride, (1,1), CuDNN.CUDNN_CROSS_CORRELATION)
      end

      bottom_offset = int(channels/layer.n_group) * height * width * sizeof(dtype)
      top_offset = int(layer.n_filter/layer.n_group) * height_out * width_out * sizeof(dtype)
      weight_offset = int(layer.n_filter/layer.n_group) * int(channels/layer.n_group) * layer.kernel[1] * layer.kernel[2] * sizeof(dtype)
      bias_offset = int(layer.n_filter/layer.n_group) * sizeof(dtype)

      etc = CuDNNConvState(inputs_desc, outputs_desc, conv_desc, filter_desc, bias_desc, 
          bottom_offset, top_offset, weight_offset, bias_offset)
    else
      error("Backend $(sys.backend) not supported")
    end

    parameters = [Parameter(filter, ∇filter, layer.filter_init, layer.filter_regu),
                  Parameter(bias, ∇bias, layer.bias_init, layer.bias_regu)]

    state = new(layer, blobs, blobs_diff, parameters)
    state.filter = filter
    state.∇filter = ∇filter
    state.bias = bias
    state.∇bias = ∇bias

    state.height_out = height_out
    state.width_out = width_out

    state.etc = etc

    return state
  end

  # Auxiliary variables
  height_out :: Int
  width_out  :: Int

  etc        :: Any # whatever status a computation backend needs to maintain
end

function setup(sys::System, layer::ConvolutionLayer, inputs::Vector{Blob})
  return ConvolutionLayerState(sys, layer, inputs)
end

function forward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob})
  channel, height, width = size(inputs[1])[2:end]
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data

    for n = 1:size(input,1)
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                output[n, (g-1)*o_g+o, y, x] = state.bias.data[(g-1)*o_g+o]

                for p = 1:state.layer.kernel[1]
                  for q = 1:state.layer.kernel[2]
                    in_y = (y-1) * state.layer.stride[1] - state.layer.pad[1] + p
                    in_x = (x-1) * state.layer.stride[2] - state.layer.pad[2] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_y <= width)
                      output[n, (g-1)*o_g+o, y, x] += input[n, (g-1)*k_g+k, in_y, in_x] *
                                                      state.filter.data[(g-1)*o_g+o, k, p, q]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end

function backward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  channel, height, width = size(inputs[1])[2:end]
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  fill!(state.∇filter.data, 0)

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs_diff[i].data

    diff = diffs[i]
    if !isa(diff, NullBlob)
      fill!(diff.data, 0)
    end

    for n = 1:size(input,1)
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                state.∇bias.data[(g-1)*o_g+o] = output[n, (g-1)*o_g+o, y, x]

                for p = 1:state.layer.kernel[1]
                  for q = 1:state.layer.kernel[2]
                    in_y = (y-1) * state.layer.stride[1] - state.layer.pad[1] + p
                    in_x = (x-1) * state.layer.stride[2] - state.layer.pad[2] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_y <= width)
                      state.∇filter.data[(g-1)*o_g+o,k,p,q] += output[n,(g-1)*o_g+o,y,x] *
                                                               input[n,(g-1)*k_g+k,in_y,in_x]
                      if !isa(diff, NullBlob)
                        diff.data[n,(g-1)*k_g+k,in_y,in_x] += output[n,(g-1)*o_g+o,y,x] *
                                                              state.filter.data[(g-1)*o_g+o,k,p,q]
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
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
