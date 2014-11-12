############################################################
# Softmax Layer
############################################################
@defstruct SoftmaxLayer CompLayer (
  name :: String = "softmax",
  tops :: Vector{Symbol} = Symbol[],
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops))
)

type SoftmaxLayerState <: LayerState
  layer      :: SoftmaxLayer
  blobs      :: Vector{Blob}

  etc        :: Any
end

type CuDNNSoftmaxState
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup(sys::System, layer::SoftmaxLayer, inputs::Vector{Blob})
  data_type  = eltype(inputs[1])
  blobs      = Blob[make_blob(sys.backend, data_type, size(input)) for input in inputs]
  etc        = nothing

  if isa(sys.backend, CuDNNBackend)
    inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
    outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
    for i = 1:length(inputs)
      inputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, size(inputs[i]))
      outputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, size(inputs[i]))
    end
    etc = CuDNNSoftmaxState(inputs_desc, outputs_desc)
  end

  state = SoftmaxLayerState(layer, blobs, etc)
  return state
end

function forward(sys::System{CPUBackend}, state::SoftmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input  = inputs[i].data
    output = state.blobs[i].data

    width, height, channels, num = size(input)

    for w = 1:width
      for h = 1:height
        for n = 1:num
          maxval = -Inf
          @simd for c = 1:channels
            @inbounds maxval = max(maxval, input[w,h,c,n])
          end
          @simd for c = 1:channels
            @inbounds output[w,h,c,n] = exp(input[w,h,c,n]-maxval)
          end
          the_sum = 0.0
          @simd for c = 1:channels
            @inbounds the_sum += output[w,h,c,n]
          end
          @simd for c = 1:channels
            @inbounds output[w,h,c,n] /= the_sum
          end
        end
      end
    end
  end
end

function forward(sys::System{CuDNNBackend}, state::SoftmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    CuDNN.softmax_forward(sys.backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
        CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, state.etc.inputs_desc[i], inputs[i].ptr,
        state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

#--------------------------------------------------------------------------------
# I'm commenting out this function as currently back propagation for CPU
# mode is not implemented. A logistic-regression like multiclass classification
# top layer should be trained by the SoftmaxLossLayer directly, instead
# of a SoftmaxLayer plus a Multiclass Logistic Loss layer. Because the
# former has much cheaper computation and beter robustness when computing
# the gradients.
#
# If somehow this function is needed in the future, e.g. training a softmax
# layer with some other loss function, blobs_diff field needs to be added
# to SoftmaxLayerState before this function could work properly.
#--------------------------------------------------------------------------------
#function backward(sys::System{CuDNNBackend}, state::SoftmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
#  for i = 1:length(inputs)
#    diff = diffs[i]
#    if isa(diff, CuTensorBlob)
#      CuDNN.softmax_backward(sys.backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
#          CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, state.etc.outputs_desc[i], state.blobs[i].ptr,
#          state.etc.outputs_desc[i], state.blobs_diff[i].ptr,
#          state.etc.inputs_desc[i], diffs[i].ptr)
#    end
#  end
#end
