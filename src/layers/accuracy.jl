@defstruct AccuracyLayer StatLayer (
  (tops :: Vector{String} = String[], length(tops) == 0),
  (bottoms :: Vector{String} = String[], length(bottoms) == 2)
)

type AccuracyLayerState <: LayerState
  layer :: AccuracyLayer
  blobs :: Vector{Blob}
end

function setup(sys::System, layer::AccuracyLayer, inputs::Vector{Blob})
  return AccuracyLayerState(layer, Blob[])
end

function forward(sys::System{CPUBackend}, layer::AccuracyLayer, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data

  accuracy = sum(int(pred) == int(label)) / length(label)
  println("  accuracy = $accuracy")
end
