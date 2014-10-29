@defstruct PoolingLayer CompLayer (
  (kernel :: Vector{Int} = [1,1], length(kernel)==2 && all(kernel .> 0)),
  (stride :: Vector{Int} = [1,1], length(stride)==2 && all(stride .> 0)),
  (pad :: Vector{Int} = [0,0], length(pad)==2 && all(pad .>= 0)),
  (bottoms :: Vector{String} = String[], length(bottoms) > 0),
  (tops :: Vector{String} = String[], length(tops) == length(bottoms)),
  pooling :: PoolingFunction = Pooling.Max()
)

type PoolingLayerState <: LayerState
  layer      :: PoolingLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(sys::System, layer::PoolingLayer, inputs::Vector{Blob})
  channels, height, width = size(inputs[1])[2:end]
  pooled_height = int(ceil(float(height+2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]))+1
  pooled_width  = int(ceil(float(width +2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]))+1

  # make sure the last pooling is not purely pooling padded area
  if any(layer.pad .> 0)
    if ((pooled_height-1) * layer.strade[1] >= height + layer.pad[1])
      pooled_height -= 1
    end
    if ((pooled_width-1) * layer.stride[2] >= width + layer.pad[2])
      pooled_width -= 1
    end
  end

  dtype = eltype(inputs[1])
  if isa(sys.backend, CPU)
    blobs = Array(CPUBlob, length(inputs))
    blobs_diff = Array(CPUBlob, length(inputs))
    if isa(layer.pooling, Pooling.Max)
      layer.pooling.masks = Array(Array, length(inputs))
    end

    for i = 1:length(inputs)
      blobs[i] = CPUBlob(Array(dtype, size(inputs[i],1), channels, pooled_height, pooled_width))
      blobs_diff[i] = CPUBlob(similar(blobs[i].data))
      if isa(layer.pooling, Pooling.Max)
        layer.pooling.masks[i] = similar(blobs[i].data, Int)
      end
    end
  else
    error("Backend $(sys.backend) not supported")
  end

  state = PoolingLayerState(layer, blobs, blobs_diff)
end

function forward(sys::System{CPU}, state::PoolingLayerState, inputs::Vector{Blob})
  forward(sys, state.layer.pooling, state, inputs)
end
function forward(sys::System{CPU}, pool::Pooling.Max, state::PoolingLayerState, inputs::Vector{Blob})
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

function backward(sys::System{CPU}, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  backward(sys, state.layer.pooling, state, inputs, diffs)
end
function backward(sys::System{CPU}, pool::Pooling.Max, state::PoolingLayerState,
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
