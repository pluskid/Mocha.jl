@defstruct CropLayer Layer (
  name :: AbstractString = "crop",
  random_crop :: Bool = false,
  random_mirror :: Bool = false,
  (crop_size :: NTuple{2, Int} = (0,0), crop_size[1] > 0 && crop_size[2] > 0),
  (bottoms :: Vector{Symbol} = [], length(bottoms) > 0),
  (tops :: Vector{Symbol} = [], length(tops) == length(bottoms))
)

type CropLayerState <: LayerState
  layer      :: CropLayer
  blobs      :: Vector{Blob}
end

function setup(backend::Backend, layer::CropLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(inputs)
    @assert ndims(inputs[i]) == 4 "Input blob $(layer.bottoms[i]) should be a 4D tensor"
    @assert isa(diffs[i], NullBlob) # Back-propagation for crop-layer is not implemented
  end

  blobs = Array{Blob}(length(inputs))
  for i = 1:length(inputs)
    width, height, channels, num = size(inputs[i])
    @assert layer.crop_size[1] <= width && layer.crop_size[2] <= height
    blobs[i] = make_blob(backend, eltype(inputs[i]),
        (layer.crop_size[1], layer.crop_size[2],channels,num))
  end

  return CropLayerState(layer, blobs)
end

function shutdown(backend::Backend, state::CropLayerState)
  map(destroy, state.blobs)
end

function crop_blob{T}(input::Array{T}, output::Array{T}, crop_size::NTuple{2,Int}, offsets::NTuple{2,Int})
  crop_w = crop_size[1]; w_off = offsets[1]
  crop_h = crop_size[2]; h_off = offsets[2]
  num = size(input, 4); channels = size(input, 3)

  for n = 1:num
    for c = 1:channels
      for h = 1:crop_h
        @simd for w = 1:crop_w
          @inbounds output[w,h,c,n] = input[w+w_off,h+h_off,c,n]
        end
      end
    end
  end
end
function mirror_crop_blob{T}(input::Array{T}, output::Array{T}, crop_size::NTuple{2,Int}, offsets::NTuple{2,Int})
  crop_w = crop_size[1]; w_off = offsets[1]
  crop_h = crop_size[2]; h_off = offsets[2]
  num = size(input, 4); channels = size(input, 3)

  for n = 1:num
    for c = 1:channels
      for h = 1:crop_h
        @simd for w = 1:crop_w
          @inbounds output[crop_w-w+1,h,c,n] = input[w+w_off,h+h_off,c,n]
        end
      end
    end
  end
end

function forward(backend::CPUBackend, state::CropLayerState, inputs::Vector{Blob})
  crop_size = state.layer.crop_size
  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data
    if state.layer.random_crop
      w_off = abs(rand(Int)) % (size(input,1)-crop_size[1]+1)
      h_off = abs(rand(Int)) % (size(input,2)-crop_size[2]+1)
    else
      w_off = div(size(input,1)-crop_size[1], 2)
      h_off = div(size(input,2)-crop_size[2], 2)
    end
    if state.layer.random_mirror && rand(UInt)%2 == 0
      mirror_crop_blob(input, output, crop_size, (w_off, h_off))
    else
      crop_blob(input, output, crop_size, (w_off, h_off))
    end
  end
end

function backward(backend::Backend, state::CropLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # backward for a crop layer could be implemented, but since crop layer is typically used
  # directly on top of a data layer, which does not need back propagation, we don't implement
  # backward for CropLayer
end
