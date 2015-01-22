@defstruct ConvolutionLayer Layer (
  (name :: String = "", !isempty(name)),
  param_key :: String = "",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: NTuple{2,Int} = (1,1), all([kernel...] .> 0)),
  (stride :: NTuple{2,Int} = (1,1), all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), all([pad...] .>= 0)),
  (n_filter :: Int = 1, n_filter > 0),
  (n_group :: Int = 1, n_group > 0),
  neuron :: ActivationFunction = Neurons.Identity(),
  filter_init :: Initializer = XavierInitializer(),
  bias_init :: Initializer = ConstantInitializer(0),
  filter_regu :: Regularizer = L2Regu(1),
  bias_regu :: Regularizer = NoRegu(),
  filter_cons :: Constraint = NoCons(),
  bias_cons :: Constraint = NoCons(),
  filter_lr :: FloatingPoint = 1.0,
  bias_lr :: FloatingPoint = 2.0,
  deconv :: Bool = false,
)
@characterize_layer(ConvolutionLayer,
  has_param  => true,
  has_neuron => true,
  can_do_bp  => true
)

type CPUConvState
  col_buffer      :: Blob

  # a all-1 vector used in gemm to help bias calculation
  bias_multiplier :: Blob
  img_buffer      :: Array
end

function setup_etc(backend::CPUBackend, layer::ConvolutionLayer, dtype, state, inputs)

  if layer.kernel[1] == 1 && layer.kernel[2] == 1 &&
     layer.stride[1] == 1 && layer.stride[2] == 1 &&
     layer.pad[1] == 0 && layer.pad[2] == 0
    col_buffer = NullBlob()
  else
    if layer.deconv
      col_buffer = CPUBlob(Array(dtype, state.width, state.height, state.kernel_dim))
    else
      col_buffer = CPUBlob(Array(dtype, state.width_out, state.height_out, state.kernel_dim))
    end
  end

  #M = height_out * width_out
  #N = div(layer.n_filter, layer.n_group)
  #K = div(channels * layer.kernel[1] * layer.kernel[2], layer.n_group)

  bias_multiplier = make_blob(backend, ones(dtype, state.width_out*state.height_out))
  if layer.deconv
    img_buffer = Array(dtype, state.width_out, state.height_out, layer.n_filter)
  else
    img_buffer = Array(dtype, state.width, state.height, state.channels)
  end
  etc = CPUConvState(col_buffer, bias_multiplier, img_buffer)
  return etc
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

  ConvolutionLayerState(backend::Backend, layer::ConvolutionLayer, shared_params, inputs::Vector{Blob}) = begin
    for i = 1:length(inputs)
      # currently we only handle 4D-tensor
      @assert ndims(inputs[i]) == 4
    end

    width, height, channels, batch_size = size(inputs[1])
    dtype = eltype(inputs[1])

    @assert channels % layer.n_group == 0
    @assert layer.n_filter % layer.n_group == 0

    state = new(layer)
    state.width      = width
    state.height     = height
    state.channels   = channels
    state.batch_size = batch_size
    state.dtype      = dtype

    if layer.deconv
      state.width_out      = layer.stride[1] * (width-1)  + layer.kernel[1] - 2*layer.pad[1]
      state.height_out     = layer.stride[2] * (height-1) + layer.kernel[2] - 2*layer.pad[2]

      state.conv_width_in  = state.width_out
      state.conv_height_in = state.height_out
      state.conv_out_sp    = width*height
      state.conv_chann_in  = layer.n_filter
      state.conv_chann_out = channels
    else
      state.width_out      = div(width  + 2*layer.pad[1]-layer.kernel[1], layer.stride[1]) + 1
      state.height_out     = div(height + 2*layer.pad[2]-layer.kernel[2], layer.stride[2]) + 1

      state.conv_width_in  = width
      state.conv_height_in = height
      state.conv_out_sp    = state.width_out*state.height_out
      state.conv_chann_in  = channels
      state.conv_chann_out = layer.n_filter
    end

    state.kernel_dim    = state.conv_chann_in * prod(layer.kernel)
    state.weight_offset = div(state.conv_chann_out * state.kernel_dim, layer.n_group*layer.n_group) * sizeof(state.dtype)
    state.col_offset    = div(state.kernel_dim * state.conv_out_sp, layer.n_group) * sizeof(state.dtype)
    state.output_offset = div(state.conv_chann_out * state.conv_out_sp, layer.n_group) * sizeof(state.dtype)

    # Make sure all input blobs are of the same shape
    for i = 2:length(inputs)
      @assert (width,height,channels,batch_size) == size(inputs[i])
      @assert dtype == eltype(inputs[i])
    end

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      blobs[i] = make_blob(backend, dtype, state.width_out, state.height_out, layer.n_filter, batch_size)
      blobs_diff[i] = make_blob(backend, dtype, state.width_out, state.height_out, layer.n_filter, batch_size)
    end

    filter_dims = tuple(layer.kernel...,div(state.conv_chann_in,layer.n_group),state.conv_chann_out)
    if shared_params != nothing
      @assert length(shared_params) == 2
      @assert shared_params[1].name == "filter" && shared_params[2].name == "bias"
      @assert size(shared_params[1].blob) == filter_dims
      @assert eltype(shared_params[1].blob) == dtype
      @debug("ConvolutionLayer($(layer.name)): sharing filters and bias")

      param_filter, param_bias = [share_parameter(backend, param) for param in shared_params]
    else
      param_filter = make_parameter(backend,"filter",dtype, filter_dims,
          layer.filter_init, layer.filter_regu, layer.filter_cons, layer.filter_lr)
      param_bias   = make_parameter(backend,"bias",dtype,(layer.n_filter,),
          layer.bias_init, layer.bias_regu, layer.bias_cons, layer.bias_lr)
    end

    filter     = param_filter.blob
    ∇filter    = param_filter.gradient
    bias       = param_bias.blob
    ∇bias      = param_bias.gradient
    parameters = [param_filter, param_bias]

    etc = setup_etc(backend, layer, dtype, state, inputs)

    state.blobs      = blobs
    state.blobs_diff = blobs_diff
    state.parameters = parameters

    state.filter     = filter
    state.∇filter    = ∇filter
    state.bias       = bias
    state.∇bias      = ∇bias

    state.etc        = etc

    state.frozen     = false

    return state
  end

  # Convolution Operation Parameters, depending on whether we are doing conv or de-conv
  conv_chann_out :: Int
  conv_chann_in  :: Int
  conv_width_in  :: Int
  conv_height_in :: Int
  conv_out_sp    :: Int

  kernel_dim     :: Int
  weight_offset  :: Int
  col_offset     :: Int
  output_offset  :: Int

  # Layer parameters, regardless of deconv or not
  height_out :: Int
  width_out  :: Int
  width      :: Int
  height     :: Int
  channels   :: Int
  batch_size :: Int
  dtype      :: Type

  etc        :: Any # whatever status a computation backend needs to maintain

  frozen     :: Bool
end

function freeze!(state::ConvolutionLayerState)
  state.frozen = true
end
function unfreeze!(state::ConvolutionLayerState)
  state.frozen = false
end
function is_frozen(state::ConvolutionLayerState)
  state.frozen
end

function setup(backend::Backend, layer::ConvolutionLayer, shared_state, inputs::Vector{Blob}, diffs::Vector{Blob})
  return ConvolutionLayerState(backend, layer, shared_state, inputs)
end
function setup(backend::Backend, layer::ConvolutionLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  setup(backend, layer, nothing, inputs, diffs)
end
function shutdown_etc(backend::CPUBackend, state::ConvolutionLayerState)
end
function shutdown(backend::Backend, state::ConvolutionLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  map(destroy, state.parameters)

  shutdown_etc(backend, state)
end

# convolution forward on one data sample
function conv_fwd_impl(backend::CPUBackend, state::ConvolutionLayerState, input::Blob, output::Blob)
  width, height, channels, num = size(input)
  img_offset = width*height*channels * sizeof(state.dtype)

  width_out, height_out, channels_out, num_out = size(output)
  top_img_offset = width_out*height_out*channels_out * sizeof(state.dtype)

  for n = 1:num
    if isa(state.etc.col_buffer, NullBlob)
      col_buffer = convert(Ptr{state.dtype}, input.data) + img_offset * (n-1)
    else
      col_buffer = state.etc.col_buffer.data
      im2col(input.data, n, col_buffer,
          width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
      col_buffer = convert(Ptr{state.dtype}, col_buffer)
    end

    output_ptr = convert(Ptr{state.dtype}, output.data) + top_img_offset * (n-1)
    for g = 1:state.layer.n_group
      RawBLAS.gemm!('N', 'N', state.conv_out_sp, div(state.conv_chann_out, state.layer.n_group),
          div(state.kernel_dim, state.layer.n_group),
          convert(state.dtype, 1), col_buffer + state.col_offset * (g-1),
          convert(Ptr{state.dtype}, pointer(state.filter.data)) + state.weight_offset * (g-1),
          convert(state.dtype, 0), output_ptr + state.output_offset * (g-1))
    end
  end
end

# convolution backward on one data sample
function conv_bwd_impl(backend::CPUBackend, state::ConvolutionLayerState, top_diff::Blob, diff::Blob)
  width, height, channels, num = size(diff)
  img_offset = width*height*channels * sizeof(state.dtype)

  width_top, height_top, channels_top, num_top = size(top_diff)
  top_img_offset = width_top*height_top*channels_top * sizeof(state.dtype)

  for n = 1:num
    top_diff_ptr = convert(Ptr{state.dtype}, top_diff.data) + top_img_offset * (n-1)
    if isa(state.etc.col_buffer, NullBlob)
      col_buffer = convert(Ptr{state.dtype}, diff.data) + img_offset * (n-1)
    else
      col_buffer = convert(Ptr{state.dtype}, state.etc.col_buffer.data)
    end

    for g = 1:state.layer.n_group
      RawBLAS.gemm!('N', 'T', state.conv_out_sp, div(state.kernel_dim, state.layer.n_group),
          div(state.conv_chann_out, state.layer.n_group),
          convert(state.dtype, 1), top_diff_ptr + state.output_offset * (g-1),
          convert(Ptr{state.dtype}, state.filter.data) + state.weight_offset * (g-1),
          convert(state.dtype, 0), col_buffer + state.col_offset * (g-1))
    end
    if !(isa(state.etc.col_buffer, NullBlob))
       col2im(state.etc.col_buffer.data, diff.data, n, state.etc.img_buffer,
          width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
    end
  end
end

function forward(backend::CPUBackend, state::ConvolutionLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    erase!(output)

    if state.layer.deconv
      conv_bwd_impl(backend, state, input, output)
    else
      conv_fwd_impl(backend, state, input, output)
    end

    # add bias
    width_out, height_out, channels_out, num_out = size(output)
    top_img_offset = width_out*height_out*channels_out * sizeof(state.dtype)

    for n = 1:num_out
      output_ptr = convert(Ptr{state.dtype}, output.data) + top_img_offset * (n-1)
      RawBLAS.gemm!('N', 'N', state.width_out*state.height_out, state.layer.n_filter, 1, convert(state.dtype, 1),
          state.etc.bias_multiplier.data, pointer(state.bias.data), convert(state.dtype, 1), output_ptr)
    end
  end
end

function backward(backend::CPUBackend, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  erase!(state.∇filter)
  erase!(state.∇bias)
  for i = 1:length(inputs)
    input = inputs[i]
    top_diff = state.blobs_diff[i]

    width, height, channels, num = size(input)
    img_offset = width*height*channels * sizeof(state.dtype)

    width_out, height_out, channels_out, num_out = size(top_diff)
    top_img_offset = height_out * width_out * channels_out * sizeof(state.dtype)

    if !state.frozen
      for n = 1:num
        top_diff_ptr = convert(Ptr{state.dtype}, top_diff.data) + top_img_offset * (n-1)
        input_ptr = convert(Ptr{state.dtype}, input.data) + img_offset * (n-1)

        #----------------------------------------------
        # bias gradient
        RawBLAS.gemv!('T', state.width_out*state.height_out, state.layer.n_filter, convert(state.dtype, 1), top_diff_ptr,
            state.etc.bias_multiplier.data, convert(state.dtype, 1), pointer(state.∇bias.data))

        #----------------------------------------------
        # filter gradient
        if state.layer.deconv
          if isa(state.etc.col_buffer, NullBlob)
            col_buffer = convert(Ptr{state.dtype}, top_diff.data) + top_img_offset * (n-1)
          else
            col_buffer = state.etc.col_buffer.data
            im2col(top_diff.data, n, col_buffer,
                width_out, height_out, channels_out, state.layer.kernel, state.layer.pad, state.layer.stride)
            col_buffer = convert(Ptr{state.dtype}, col_buffer)
          end
          for g = 1:state.layer.n_group
            RawBLAS.gemm!('T', 'N',
                div(state.kernel_dim, state.layer.n_group),
                div(state.conv_chann_out, state.layer.n_group),
                state.conv_out_sp,
                one(state.dtype),
                col_buffer + state.col_offset * (g-1),
                input_ptr + state.output_offset * (g-1),
                one(state.dtype),
                convert(Ptr{state.dtype}, pointer(state.∇filter.data)) + state.weight_offset * (g-1))
          end
        else
          if isa(state.etc.col_buffer, NullBlob)
            col_buffer = convert(Ptr{state.dtype}, input.data) + img_offset * (n-1)
          else
            col_buffer = state.etc.col_buffer.data
            im2col(input.data, n, col_buffer,
                width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
            col_buffer = convert(Ptr{state.dtype}, col_buffer)
          end
          for g = 1:state.layer.n_group
            RawBLAS.gemm!('T', 'N',
                div(state.kernel_dim, state.layer.n_group),
                div(state.conv_chann_out, state.layer.n_group),
                state.conv_out_sp, convert(state.dtype, 1),
                col_buffer + state.col_offset * (g-1),
                top_diff_ptr + state.output_offset * (g-1),
                one(state.dtype),
                convert(Ptr{state.dtype}, pointer(state.∇filter.data)) + state.weight_offset * (g-1))
          end
        end
      end
    end

    #----------------------------------------------
    # back propagate gradient
    if isa(diffs[i], CPUBlob)
      diff = diffs[i]
      if state.layer.deconv
        conv_fwd_impl(backend, state, top_diff, diff)
      else
        conv_bwd_impl(backend, state, top_diff, diff)
      end
    end
  end
end

