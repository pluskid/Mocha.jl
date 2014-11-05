@defstruct PoolingLayer CompLayer (
  (kernel :: NTuple{2, Int} = (1,1), length(kernel)==2 && all([kernel...] .> 0)),
  (stride :: NTuple{2, Int} = (1,1), length(stride)==2 && all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = [0,0], length(pad)==2 && all([pad...] .>= 0)),
  (bottoms :: Vector{String} = String[], length(bottoms) > 0),
  (tops :: Vector{String} = String[], length(tops) == length(bottoms)),
  pooling :: PoolingFunction = Pooling.Max()
)

type PoolingLayerState <: LayerState
  layer      :: PoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  etc        :: Any
end

type CuDNNPoolingState
  pooling_desc :: CuDNN.PoolingDescriptor
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup(sys::System, layer::PoolingLayer, inputs::Vector{Blob})
  width    = get_width(inputs[1])
  height   = get_height(inputs[1])

  pooled_width  = int(ceil(float(width +2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]))+1
  pooled_height = int(ceil(float(height+2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]))+1

  # make sure the last pooling is not purely pooling padded area
  if layer.pad[1] > 0 || layer.pad[2] > 0
    if ((pooled_height-1) * layer.strade[2] >= height + layer.pad[2])
      pooled_height -= 1
    end
    if ((pooled_width-1) * layer.stride[1] >= width + layer.pad[1])
      pooled_width -= 1
    end
  end

  dtype = eltype(inputs[1])
  blobs = Array(Blob, length(inputs))
  blobs_diff = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    blobs[i] = make_blob(sys.backend,dtype,
        (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
    blobs_diff[i] = make_blob(sys.backend,dtype,
        (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
  end

  if isa(sys.backend, CPUBackend) 
    error("TODO")
    if isa(layer.pooling, Pooling.Max)
      layer.pooling.masks = Array(Blob, length(inputs))
      for i = 1:length(inputs)
        layer.pooling.masks[i] = Array(Int, size(blobs[i]))
      end
    end
  elseif isa(sys.backend, CuDNNBackend)
    if layer.pad[1] == 0 && layer.pad[2] == 0
      if isa(layer.pooling, Pooling.Max)
        pooling_mode = CuDNN.CUDNN_POOLING_MAX
      elseif isa(layer.pooling, Pooling.Mean)
        pooling_mode = CuDNN.CUDNN_POOLING_AVERAGE
      else
        error("TODO: pooling mode $(layer.pooling) not supported by CuDNN")
      end

      pooling_desc = CuDNN.create_pooling_descriptor(pooling_mode, layer.kernel, layer.stride)
      inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      for i = 1:length(inputs)
        inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, 
            (width,height,get_chann(inputs[i]),get_num(inputs[i])))
        outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype, 
            (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
      end
      etc = CuDNNPoolingState(pooling_desc, inputs_desc, outputs_desc)
    else
      error("TODO: CuDNN does not support pooling with padding")
    end
  end

  state = PoolingLayerState(layer, blobs, blobs_diff, etc)
end

function forward(sys::System{CPUBackend}, state::PoolingLayerState, inputs::Vector{Blob})
  forward(sys, state.layer.pooling, state, inputs)
end
function forward(sys::System{CPUBackend}, pool::Pooling.Max, state::PoolingLayerState, inputs::Vector{Blob})
  channels, height, width = size(inputs[1])[2:end]
  pooled_height, pooled_width = size(state.blobs[1])[3:end]

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data
    mask = pool.masks[i]

    fill!(output, -Inf)
    for n = 1:size(input,1)
      for c = 1:channels
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            hstart = max(1, (ph-1)*state.layer.stride[1] - state.layer.pad[1] + 1)
            wstart = max(1, (pw-1)*state.layer.stride[2] - state.layer.pad[2] + 1)
            hend = min(hstart + state.layer.kernel[1] - 1, height)
            wend = min(wstart + state.layer.kernel[2] - 1, width)

            for h = hstart:hend
              for w = wstart:wend
                if input[n, c, h, w] > output[n, c, ph, pw]
                  output[n, c, ph, pw] = input[n, c, h, w]
                  mask[n, c, ph, pw] = h*width + w
                end
              end
            end
          end
        end
      end
    end
  end
end

function backward(sys::System{CPUBackend}, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(sys, state.layer.pooling, state, inputs, diffs)
end
function backward(sys::System{CPUBackend}, pool::Pooling.Max, state::PoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  channels, height, width = size(inputs[1])[2:end]
  pooled_height, pooled_width = size(state.blobs[1])[3:end]

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      fill!(diff.data, 0)
    else
      continue # nothing to do if not propagating back
    end

    output = state.blobs_diff[i].data
    mask = pool.masks[i]

    for n = 1:size(output,1)
      for c = 1:channels
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            index = mask[n,c,ph,pw]
            w = index % width
            h = int(index / width)

            diff.data[n,c,h,w] += output[n,c,ph,pw]
          end
        end
      end
    end
  end
end

function forward(sys::System{CuDNNBackend}, state::PoolingLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    CuDNN.pooling_forward(sys.backend.cudnn_ctx, state.etc.pooling_desc, 
        state.etc.inputs_desc[i], inputs[i].ptr,
        state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

function backward(sys::System{CuDNNBackend}, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(inputs)
    if isa(diffs[i], CuTensorBlob)
      CuDNN.pooling_backward(sys.backend.cudnn_ctx, state.etc.pooling_desc,
          state.etc.outputs_desc[i], state.blobs[i].ptr,
          state.etc.outputs_desc[i], state.blobs_diff[i].ptr,
          state.etc.inputs_desc[i], inputs[i].ptr,
          state.etc.inputs_desc[i], diffs[i].ptr)
    end
  end
end

