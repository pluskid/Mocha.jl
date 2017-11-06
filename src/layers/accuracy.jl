@defstruct AccuracyLayer Layer (
  name :: AbstractString = "accuracy",
  report_error :: Bool = false,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(AccuracyLayer,
  is_sink    => true,
  has_stats  => true,
)

type AccuracyLayerState <: LayerState
  layer :: AccuracyLayer

  op_dim   :: Int
  accuracy :: Float64
  n_accum  :: Int
  etc      :: Any
end

function setup_etc(backend::CPUBackend, layer::AccuracyLayer, op_dim::Int, inputs)
  nothing
end

function setup(backend::Backend, layer::AccuracyLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  total_dim = ndims(inputs[1])
  dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
  @assert 1 <= dim <= total_dim
  @assert dim != total_dim

  etc = setup_etc(backend, layer, dim, inputs)
  return AccuracyLayerState(layer, dim, 0.0, 0, etc)
end
function shutdown(backend::CPUBackend, state::AccuracyLayerState)
end

function reset_statistics(state::AccuracyLayerState)
  state.n_accum = 0
  state.accuracy = 0.0
end

function dump_statistics(storage, state::AccuracyLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-accuracy", state.accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-state.accuracy)
  end

  if show
    accuracy = @sprintf("%.4f%%", state.accuracy*100)
    m_info("  Accuracy (avg over $(state.n_accum)) = $accuracy")
  end
end

function forward(backend::CPUBackend, state::AccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data

  dim_pre, dim_prob, dim_post = split_dims(pred, state.op_dim)

  accuracy = 0.0
  for i = 0:dim_pre-1
    for j = 0:dim_post-1
      idx = Int[i + dim_pre*(k + dim_prob*j) for k=0:dim_prob-1] + 1
      @inbounds if round(Int, label[i + dim_pre*j + 1])+1 == indmax(pred[idx])
        accuracy += 1.0
      end
    end
  end

  state.accuracy = convert(Float64, state.accuracy * state.n_accum + accuracy) / (state.n_accum + length(label))
  state.n_accum += length(label)
end

function backward(backend::Backend, state::AccuracyLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

