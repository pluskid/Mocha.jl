@defstruct InnerProductLayer Layer (
  (name :: String = "", !isempty(name)),
  param_key :: String = "",
  (tops :: Vector{Symbol} = Symbol[], length(tops) >= 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
  (output_dim :: Int = 0, output_dim > 0),
  weight_init :: Initializer = XavierInitializer(),
  bias_init :: Initializer = ConstantInitializer(0),
  weight_regu :: Regularizer = L2Regu(1),
  bias_regu :: Regularizer = NoRegu(),
  weight_cons :: Constraint = NoCons(),
  bias_cons :: Constraint = NoCons(),
  weight_lr :: FloatingPoint = 1.0,
  bias_lr :: FloatingPoint = 2.0,
  neuron :: ActivationFunction = Neurons.Identity(),
  recurrent :: Bool = false;
)
@characterize_layer(InnerProductLayer,
  can_do_bp  => true,
  has_param  => true,
  has_neuron => true
)

type InnerProductLayerState <: LayerState
  layer      :: InnerProductLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  W  :: Blob
  ∇W :: Blob
  b  :: Blob
  ∇b :: Blob

  # a all-1 vector used in gemm to help bias calculation
  bias_multipliers :: Vector{Blob}

  frozen :: Bool

  InnerProductLayerState(backend::Backend, layer::InnerProductLayer, shared_params, inputs::Vector{Blob}) = begin
    fea_size = get_fea_size(inputs[1])
    data_type = eltype(inputs[1])
    out_dim = layer.output_dim

    # make sure all blobs has the same feature dimension (batch_size could be different)
    for i = 2:length(inputs)
      @assert get_fea_size(inputs[i]) == fea_size
      @assert eltype(inputs[i]) == data_type
    end

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))
    bias_multipliers = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      nums = get_num(inputs[i])
      blobs[i] = make_blob(backend, data_type, out_dim, nums)
      blobs_diff[i] = make_blob(backend, data_type, out_dim, nums)
      bias_multipliers[i] = make_blob(backend, ones(data_type, 1, nums))
    end

    state = new(layer, blobs, blobs_diff)
    state.bias_multipliers = bias_multipliers

    if shared_params != nothing
      @assert length(shared_params) == 2
      @assert shared_params[1].name == "weight" && shared_params[2].name == "bias"
      @assert size(shared_params[1].blob) == (fea_size, out_dim)
      @assert eltype(shared_params[1].blob) == data_type
      @assert size(shared_params[2].blob) == (out_dim, 1)
      @debug("InnerProductLayer($(layer.name)): sharing weights and bias")

      param_weight, param_bias = [share_parameter(backend, param) for param in shared_params]
    else
      param_weight = make_parameter(backend, "weight", data_type, (fea_size,out_dim),
          layer.weight_init, layer.weight_regu, layer.weight_cons, layer.weight_lr)
      param_bias   = make_parameter(backend, "bias", data_type, (out_dim,1),
          layer.bias_init, layer.bias_regu, layer.bias_cons, layer.bias_lr)
    end

    state.W  = param_weight.blob
    state.∇W = param_weight.gradient
    state.b  = param_bias.blob
    state.∇b = param_bias.gradient
    state.parameters = [param_weight, param_bias]
    state.frozen = false

    return state
  end
end

function freeze!(state::InnerProductLayerState)
  state.frozen = true
end
function unfreeze!(state::InnerProductLayerState)
  state.frozen = false
end
function is_frozen(state::InnerProductLayerState)
  state.frozen
end

function setup(backend::Backend, layer::InnerProductLayer, shared_state, inputs::Vector{Blob}, diffs::Vector{Blob})
  state = InnerProductLayerState(backend, layer, shared_state, inputs)
  return state
end
function setup(backend::Backend, layer::InnerProductLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  setup(backend, layer, nothing, inputs, diffs)
end
function shutdown(backend::Backend, state::InnerProductLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  map(destroy, state.bias_multipliers)
  map(destroy, state.parameters)
end

function forward(backend::CPUBackend, state::InnerProductLayerState, inputs::Vector{Blob})
  M = size(state.W, 2)   # target dim
  K = size(state.W, 1)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    N = get_num(input)   # batch size
    output = state.blobs[i]
    # output = W^T * X
    BLAS.gemm!('T', 'N', one(dtype), state.W.data,
                reshape(input.data, (K,N)), zero(dtype), output.data)
    # output += bias
    BLAS.gemm!('N', 'N', one(dtype), state.b.data,
                state.bias_multipliers[i].data, one(dtype), output.data)
  end
end

function backward(backend::CPUBackend, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  target_dim = size(state.W, 2)
  source_dim = size(state.W, 1)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = zero(data_type)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    batch_size = get_num(input)
    ∂f_∂o = state.blobs_diff[i]

    if !state.frozen
      BLAS.gemm!('N', 'T', one(data_type), reshape(input.data, (source_dim, batch_size)),
                 ∂f_∂o.data, zero_and_then_one, state.∇W.data)

      # ∂f/∂b = sum(∂f/∂o, 2)
      BLAS.gemm!('N', 'N', one(data_type), ∂f_∂o.data,
                 reshape(state.bias_multipliers[i].data, (batch_size, 1)),
                 zero_and_then_one, state.∇b.data)
    end

    zero_and_then_one = one(data_type)

    # if back propagate down
    if isa(diffs[i], CPUBlob)
      # ∂f/∂x = W * [∂f/∂o]
      BLAS.gemm!('N', 'N', one(data_type), state.W.data,
                 ∂f_∂o.data, zero(data_type),
                 reshape(diffs[i].data, (source_dim, batch_size)))
    end
  end
end

