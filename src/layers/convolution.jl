@defstruct ConvolutionLayer CompLayer (
  name :: String = "convolution",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: NTuple{2,Int} = (1,1), length(kernel)==2 && all([kernel...] .> 0)),
  (stride :: NTuple{2,Int} = (1,1), length(stride)==2 && all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), length(pad)==2 && all([pad...] .>= 0)),
  (n_filter :: Int = 1, n_filter > 0),
  (n_group :: Int = 1, n_group > 0),
  neuron :: ActivationFunction = Neurons.Identity(),
  filter_init :: Initializer = XavierInitializer(),
  bias_init :: Initializer = ConstantInitializer(0),
  filter_regu :: Regularizer = L2Regu(1),
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

type CPUConvState
  col_buffer      :: Blob
  M               :: Int
  N               :: Int
  K               :: Int

  # a all-1 vector used in gemm to help bias calculation
  bias_multiplier :: Blob
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

  ConvolutionLayerState(sys::System, layer::ConvolutionLayer, shared_state, inputs::Vector{Blob}) = begin
    channels = get_chann(inputs[1])
    @assert channels % layer.n_group == 0
    @assert layer.n_filter % layer.n_group == 0

    batch_size = get_num(inputs[1])
    height = get_height(inputs[1])
    width  = get_width(inputs[1])
    width_out  = floorint((width  + 2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]) + 1
    height_out = floorint((height + 2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]) + 1

    dtype = eltype(inputs[1])

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      blobs[i] = make_blob(sys.backend, dtype, width_out, height_out, layer.n_filter, batch_size)
      blobs_diff[i] = make_blob(sys.backend, dtype, width_out, height_out, layer.n_filter, batch_size)
    end

    if isa(shared_state, ConvolutionLayerState)
      @assert size(shared_state.filter) == tuple(layer.kernel...,floorint(channels/layer.n_group),layer.n_filter)
      @assert eltype(shared_state.filter) == dtype
      @debug("Sharing filters and bias with an existing ConvolutionLayer")

      filter = shared_state.filter
      ∇filter = shared_state.∇filter
      bias = shared_state.bias
      ∇bias = shared_state.bias
    else
      filter = make_blob(sys.backend, dtype, layer.kernel[1], layer.kernel[2],
          floorint(channels/layer.n_group), layer.n_filter)
      ∇filter = make_blob(sys.backend, dtype, layer.kernel[1], layer.kernel[2],
          floorint(channels/layer.n_group), layer.n_filter)
      bias = make_blob(sys.backend, dtype, layer.n_filter)
      ∇bias = make_blob(sys.backend, dtype, layer.n_filter)
    end

    if isa(sys.backend, CPUBackend)
      if layer.kernel[1] == 1 && layer.kernel[2] == 1 &&
         layer.stride[1] == 1 && layer.stride[2] == 1 &&
         layer.pad[1] == 0 && layer.pad[2] == 0
        col_buffer = NullBlob()
      else
        col_buffer = CPUBlob(Array(dtype, width_out, height_out, channels*prod(layer.kernel), 1))
      end
      M = height_out * width_out
      N = floorint(layer.n_filter / layer.n_group)
      K = floorint(channels * layer.kernel[1] * layer.kernel[2] / layer.n_group)
      bias_multiplier = CPUBlob(dtype, M)
      fill!(bias_multiplier, convert(dtype,1))
      etc = CPUConvState(col_buffer, M, N, K, bias_multiplier)
    elseif isa(sys.backend, CuDNNBackend)
      filter_desc = CuDNN.create_filter_descriptor(dtype, (layer.kernel[1], layer.kernel[2],
          floorint(channels/layer.n_group), floorint(layer.n_filter/layer.n_group)))
      bias_desc = CuDNN.create_tensor4d_descriptor(dtype, (1,1,floorint(layer.n_filter/layer.n_group),1))

      inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      conv_desc = Array(CuDNN.ConvolutionDescriptor, length(inputs))
      for i = 1:length(inputs)
        inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width,height,floorint(channels/layer.n_group),batch_size),
            (1, width, width*height, width*height*channels))
        outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, (width_out, height_out, floorint(layer.n_filter/layer.n_group), batch_size),
            (1, width_out, width_out*height_out, width_out*height_out*layer.n_filter))
        conv_desc[i] = CuDNN.create_convolution_descriptor(inputs_desc[i], filter_desc, layer.pad,
            layer.stride, (1,1), CuDNN.CUDNN_CROSS_CORRELATION)
      end

      bottom_offset = floorint(channels/layer.n_group) * height * width * sizeof(dtype)
      top_offset = floorint(layer.n_filter/layer.n_group) * height_out * width_out * sizeof(dtype)
      weight_offset = floorint(layer.n_filter/layer.n_group) * floorint(channels/layer.n_group) * layer.kernel[1] * layer.kernel[2] * sizeof(dtype)
      bias_offset = floorint(layer.n_filter/layer.n_group) * sizeof(dtype)

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

function setup(sys::System, layer::ConvolutionLayer, shared_state, inputs::Vector{Blob})
  return ConvolutionLayerState(sys, layer, shared_state, inputs)
end

function forward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    erase!(output)
    dtype = eltype(input)
    width, height, channels, num = size(input)
    img_offset = width*height*channels * sizeof(dtype)
    weight_offset = state.etc.N * state.etc.K * sizeof(dtype)
    col_offset = state.etc.M * state.etc.K * sizeof(dtype)
    top_offset = state.etc.M * state.etc.N * sizeof(dtype)
    top_img_offset = state.height_out * state.width_out * state.layer.n_filter * sizeof(dtype)

    for n = 1:num
      if isa(state.etc.col_buffer, NullBlob)
        col_buffer = convert(Ptr{dtype}, input.data) + img_offset * (n-1)
      else
        col_buffer = state.etc.col_buffer.data
        im2col(sub(input.data, 1:width, 1:height, 1:channels, n), col_buffer,
            width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
        col_buffer = convert(Ptr{dtype}, col_buffer)
      end

      output_ptr = convert(Ptr{dtype}, output.data) + top_img_offset * (n-1)
      for g = 1:state.layer.n_group
        RawBLAS.gemm!('N', 'N', state.etc.M, state.etc.N, state.etc.K, convert(dtype, 1),
            col_buffer + col_offset * (g-1),
            convert(Ptr{dtype}, state.filter.data) + weight_offset * (g-1),
            convert(dtype, 0), output_ptr + top_offset * (g-1))
      end
      RawBLAS.gemm!('N', 'N', state.etc.M, state.layer.n_filter, 1, convert(dtype, 1),
          state.etc.bias_multiplier.data, state.bias.data, convert(dtype, 1), output_ptr)
    end
  end
end

function backward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  erase!(state.∇filter)
  erase!(state.∇bias)
  for i = 1:length(inputs)
    input = inputs[i]
    top_diff = state.blobs_diff[i]
    dtype = eltype(input)
    width, height, channels, num = size(input)
    img_offset = width*height*channels * sizeof(dtype)
    weight_offset = state.etc.N * state.etc.K * sizeof(dtype)
    col_offset = state.etc.M * state.etc.K * sizeof(dtype)
    top_offset = state.etc.M * state.etc.N * sizeof(dtype)
    top_img_offset = state.height_out * state.width_out * state.layer.n_filter * sizeof(dtype)

    for n = 1:num
      top_diff_ptr = convert(Ptr{dtype}, top_diff.data) + top_img_offset * (n-1)

      #----------------------------------------------
      # bias gradient
      RawBLAS.gemv!('T', state.etc.M, state.layer.n_filter, convert(dtype, 1), top_diff_ptr, 
          state.etc.bias_multiplier.data, convert(dtype, 1), state.∇bias.data)

      #----------------------------------------------
      # filter gradient
      if isa(state.etc.col_buffer, NullBlob)
        col_buffer = convert(Ptr{dtype}, input.data) + img_offset * (n-1)
      else
        col_buffer = state.etc.col_buffer.data
        im2col(sub(input.data, 1:width, 1:height, 1:channels, n), col_buffer,
            width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
        col_buffer = convert(Ptr{dtype}, col_buffer)
      end
      for g = 1:state.layer.n_group
        RawBLAS.gemm!('T', 'N', state.etc.K, state.etc.N, state.etc.M, convert(dtype, 1),
            col_buffer + col_offset * (g-1),
            top_diff_ptr + top_offset * (g-1), convert(dtype, 1), 
            convert(Ptr{dtype}, state.∇filter.data) + weight_offset * (g-1))
      end
    end

    #----------------------------------------------
    # back propagate gradient
    if isa(diffs[i], CPUBlob)
      diff = diffs[i]
      for n = 1:num
        top_diff_ptr = convert(Ptr{dtype}, top_diff.data) + top_img_offset * (n-1)
        if isa(state.etc.col_buffer, NullBlob)
          col_buffer = convert(Ptr{dtype}, diff.data) + img_offset * (n-1)
        else
          col_buffer = convert(Ptr{dtype}, state.etc.col_buffer.data)
        end

        for g = 1:state.layer.n_group
          RawBLAS.gemm!('N', 'T', state.etc.M, state.etc.K, state.etc.N, convert(dtype, 1),
              top_diff_ptr + top_offset * (g-1),
              convert(Ptr{dtype}, state.filter.data) + weight_offset * (g-1),
              convert(dtype, 0), col_buffer + col_offset * (g-1))
        end
        if !(isa(state.etc.col_buffer, NullBlob))
          col2im(state.etc.col_buffer.data, sub(diff.data, 1:width, 1:height, 1:channels, n),
              width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
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
