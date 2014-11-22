export LRNModeType, LRNMode

abstract LRNModeType
module LRNMode
import ..LRNModeType
type AcrossChannel <: LRNModeType end
type WithinChannel <: LRNModeType end
end # module LRNMode

################################################################################
# Local Response Normalization Layer
################################################################################
@defstruct LRNLayer CompLayer (
  name :: String = "lrn",
  (kernel :: Int = 5, kernel > 0),
  (scale :: Number = 1, isreal(scale)),
  (shift :: Number = 1, isreal(shift)),
  (power :: Number = 0.75, isreal(power)),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 1),
  mode :: LRNModeType = LRNMode.AcrossChannel(),
)

type LRNLayerState <: LayerState
  layer      :: LRNLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  do_split   :: LayerState
  do_square  :: LayerState
  do_pool    :: LayerState
  do_power   :: LayerState
  do_div     :: LayerState
end

function setup(sys::System, layer::LRNLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  split_layer = SplitLayer(tops=Array(Symbol,2), bottoms=Array(Symbol,1))
  do_split = setup(sys, split_layer, inputs, diffs)

  square_layer = PowerLayer(power=2, tops=Array(Symbol,1), bottoms=Array(Symbol,1))
  do_square = setup(sys, square_layer,
      Blob[do_split.blobs[1]], Blob[do_split.blobs_diff[1]])

  pre_pad = div(layer.kernel-1,2)
  if isa(layer.mode, LRNMode.AcrossChannel)
    post_pad = layer.kernel - pre_pad - 1
    pool_layer = ChannelPoolingLayer(tops=Array(Symbol,1), bottoms=Array(Symbol,1),
        kernel=layer.kernel, stride=1, pad=(pre_pad,post_pad),
        pooling=Pooling.Mean())
  elseif isa(layer.mode, LRNMode.WithinChannel)
    pool_layer = PoolingLayer(tops=Array(Symbol,1), bottoms=Array(Symbol,1),
        kernel=(layer.kernel,layer.kernel), stride=(1,1), pad=(pre_pad,pre_pad),
        pooling=Pooling.Mean())
  else
    error("LRN-mode $(layer.mode) not supported")
  end
  do_pool = setup(sys, pool_layer, do_square.blobs, do_square.blobs_diff)

  power_layer = PowerLayer(tops=Array(Symbol,1), bottoms=Array(Symbol,1),
      power=layer.power, scale=layer.scale, shift=layer.shift)
  do_power = setup(sys, power_layer, do_pool.blobs, do_pool.blobs_diff)

  div_layer = ElementWiseLayer(tops=Array(Symbol,1), bottoms=Array(Symbol,2),
      operation = ElementWiseFunctors.Divide())
  do_div = setup(sys, div_layer,
      Blob[do_split.blobs[2],do_power.blobs[1]],
      Blob[do_split.blobs_diff[2],do_power.blobs_diff[1]])

  state = LRNLayerState(layer, do_div.blobs, do_div.blobs_diff,
      do_split, do_square, do_pool, do_power, do_div)
end
function shutdown(sys::System, state::LRNLayerState)
  shutdown(sys, state.do_split)
  shutdown(sys, state.do_square)
  shutdown(sys, state.do_pool)
  shutdown(sys, state.do_power)
  shutdown(sys, state.do_div)
end

function forward(sys::System, state::LRNLayerState, inputs::Vector{Blob})
  forward(sys, state.do_split, inputs)
  forward(sys, state.do_square, Blob[state.do_split.blobs[1]])
  forward(sys, state.do_pool, state.do_square.blobs)
  forward(sys, state.do_power, state.do_pool.blobs)
  forward(sys, state.do_div, Blob[state.do_split.blobs[2],state.do_power.blobs[1]])
end

function backward(sys::System, state::LRNLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    backward(sys, state.do_div,
        Blob[state.do_split.blobs[2],state.do_power.blobs[1]],
        Blob[state.do_split.blobs_diff[2],state.do_power.blobs_diff[1]])
    backward(sys, state.do_power, state.do_pool.blobs, state.do_pool.blobs_diff)
    backward(sys, state.do_pool, state.do_square.blobs, state.do_square.blobs_diff)
    backward(sys, state.do_square,
        Blob[state.do_split.blobs[1]],
        Blob[state.do_split.blobs_diff[1]])
    backward(sys, state.do_split, inputs, diffs)
  end
end

