@defstruct PoolingLayer CompLayer (
  name :: String = "pooling",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: NTuple{2, Int} = (1,1), length(kernel)==2 && all([kernel...] .> 0)),
  (stride :: NTuple{2, Int} = (1,1), length(stride)==2 && all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), length(pad)==2 && all([pad...] .>= 0)),
  pooling :: PoolingFunction = Pooling.Max()
)

type PoolingLayerState <: LayerState
  layer      :: PoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  etc        :: Any
end

function setup_etc(sys::System{CPUBackend}, layer::PoolingLayer, inputs,
    pooled_width, pooled_height)

  if isa(layer.pooling, Pooling.Max)
    masks = Array(Array, length(inputs))
    for i = 1:length(inputs)
      masks[i] = Array(Int, pooled_width, pooled_height, get_chann(inputs[i]), get_num(inputs[i]))
    end
    etc = masks
  else
    etc = nothing
  end
  return etc
end

function setup(sys::System, layer::PoolingLayer, inputs::Vector{Blob})
  width    = get_width(inputs[1])
  height   = get_height(inputs[1])

  pooled_width  = int(ceil(float(width +2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]))+1
  pooled_height = int(ceil(float(height+2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]))+1

  # make sure the last pooling is not purely pooling padded area
  if layer.pad[1] > 0 || layer.pad[2] > 0
    if ((pooled_height-1) * layer.stride[2] >= height + layer.pad[2])
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

  etc = setup_etc(sys, layer, inputs, pooled_width, pooled_height)
  state = PoolingLayerState(layer, blobs, blobs_diff, etc)
end

function forward(sys::System{CPUBackend}, state::PoolingLayerState, inputs::Vector{Blob})
  forward(sys, state.layer.pooling, state, inputs)
end

function forward(sys::System{CPUBackend}, pool::PoolingFunction, state::PoolingLayerState, inputs::Vector{Blob})
  width, height, channels, num = size(inputs[1])
  pooled_width = get_width(state.blobs[1])
  pooled_height = get_height(state.blobs[1])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data

    for n = 1:num
      for c = 1:channels
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            hstart = max(1, (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1)
            wstart = max(1, (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1)
            hend = min(hstart + state.layer.kernel[2] - 1, height)
            wend = min(wstart + state.layer.kernel[1] - 1, width)
            if isa(pool, Pooling.Max)
              maxval = -Inf
              maxw = 0
              maxh = 0
              for w = wstart:wend
                for h = hstart:hend
                  @inbounds val = input[w,h,c,n]
                  if val > maxval
                    maxval = val
                    maxw = w
                    maxh = h
                  end
                end
              end
              @inbounds output[pw,ph,c,n] = maxval
              @inbounds state.etc[i][pw,ph,c,n] = (maxh-1) * width + maxw-1
            elseif isa(pool, Pooling.Mean)
              the_sum = 0.0
              for w = wstart:wend
                for h = hstart:hend
                  @inbounds the_sum += input[w,h,c,n]
                end
              end
              @inbounds output[pw,ph,c,n] = the_sum / kernel_size
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

function backward(sys::System{CPUBackend}, pool::PoolingFunction, state::PoolingLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  width, height, channels, num = size(inputs[1])
  pooled_width = get_width(state.blobs[1])
  pooled_height = get_height(state.blobs[1])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      diff = diff.data
      fill!(diff, 0)
    else
      continue # nothing to do if not propagating back
    end
    top_diff = state.blobs_diff[i].data

    for n = 1:num
      for c = 1:channels
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            if isa(pool, Pooling.Max)
              index = state.etc[i][pw,ph,c,n]
              idx_w = (index % width) + 1
              idx_h = div(index, width) + 1
              @inbounds diff[idx_w, idx_h, c, n] += top_diff[pw,ph,c,n]
            elseif isa(pool, Pooling.Mean)
              hstart = max(1, (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1)
              wstart = max(1, (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1)
              hend = min(hstart + state.layer.kernel[2] - 1, height)
              wend = min(wstart + state.layer.kernel[1] - 1, width)

              @inbounds val = top_diff[pw,ph,c,n] / kernel_size
              for w = wstart:wend
                for h = hstart:hend
                  @inbounds diff[w,h,c,n] += val
                end
              end
            end
          end
        end
      end
    end
  end
end

