@defstruct AccuracyLayer StatLayer (
  name :: String = "accuracy",
  report_error :: Bool = false,
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)

type AccuracyLayerState <: LayerState
  layer :: AccuracyLayer

  accuracy :: Float64
  n_accum  :: Int
  etc      :: Any
end

function setup_etc(sys::System{CPUBackend}, layer::AccuracyLayer, inputs)
  nothing
end

function setup(sys::System, layer::AccuracyLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  etc = setup_etc(sys, layer, inputs)
  return AccuracyLayerState(layer, 0.0, 0, etc)
end
function shutdown(sys::System{CPUBackend}, state::AccuracyLayerState)
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
    @info("  Accuracy (avg over $(state.n_accum)) = $accuracy")
  end
end

function forward(sys::System{CPUBackend}, state::AccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data

  width, height, channels, num = size(pred)
  accuracy = 0.0
  for w = 1:width
    for h = 1:height
      for n = 1:num
        if int(label[w,h,1,n])+1 == indmax(pred[w,h,:,n])
          accuracy += 1.0
        end
      end
    end
  end
  state.accuracy = float64(state.accuracy * state.n_accum + accuracy) / (state.n_accum + length(label))
  state.n_accum += length(label)
end

function forward_and_reset_diff(sys::System{CPUBackend}, state::AccuracyLayerState, inputs::Vector{Blob})
  forward(sys, state, inputs)
end

function backward(sys::System, state::AccuracyLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

