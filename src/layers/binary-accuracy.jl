@defstruct BinaryAccuracyLayer Layer (
  name :: AbstractString = "binary-accuracy",
  report_error :: Bool = false,
  (threshold :: Float64 = 0.0, isfinite(threshold)),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(BinaryAccuracyLayer,
  is_sink    => true,
  has_stats  => true,
)

type BinaryAccuracyLayerState <: LayerState
  layer :: BinaryAccuracyLayer

  accuracy :: Float64
  n_wrong :: Int
  n_accum  :: Int

  etc :: Any
end

function setup_etc(backend::CPUBackend, layer::BinaryAccuracyLayer, inputs)
  nothing
end

function setup(backend::Backend, layer::BinaryAccuracyLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  etc = setup_etc(backend, layer, inputs)

  return BinaryAccuracyLayerState(layer, 0.0, 0, 0, etc)
end
function shutdown(backend::CPUBackend, state::BinaryAccuracyLayerState)
end

function reset_statistics(state::BinaryAccuracyLayerState)
  state.n_accum = 0
  state.n_wrong = 0
  state.accuracy = 0.0
end

function dump_statistics(storage, state::BinaryAccuracyLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-accuracy", state.accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-state.accuracy)
  end

  if show
    accuracy = @sprintf("%.4f%%", state.accuracy*100)
    @info("  Accuracy (avg over $(state.n_accum)) = $accuracy")
  end
end

function forward(backend::CPUBackend, state::BinaryAccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data

  T = eltype(pred)
  threshold = convert(T, state.layer.threshold)

  for i=1:length(pred)
    if (pred[i]>threshold) $ (label[i]>threshold)
      state.n_wrong += 1
    end
  end
  state.n_accum += length(pred)
  state.accuracy = (state.n_accum-state.n_wrong) / state.n_accum
end

function backward(backend::Backend, state::BinaryAccuracyLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

