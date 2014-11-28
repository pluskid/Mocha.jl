################################################################################
# A simple image classifier wrapper of Mocha. This is an abstracted base used by
# both image-classifier.jl and pycall-image-classifier.jl.
################################################################################

using Mocha

abstract ImageType
function imread{T <: ImageType}(::Type{T}, filename::String)
  error("Not implemented")
end

type ImageClassifier{T <: ImageType}
  net           :: Net
  channel_order :: NTuple{3,Int} # The channel order of the trained net, (1,2,3) means RGB
  sp_order      :: NTuple{2,Int} # The spatial order (1,2) means width-height (row major)
  grayscale     :: Bool          # Whether we are working with gray (single-channel) images
  classes       :: Vector        # Names for each class

  data_layer :: MemoryDataLayer
  pred_blob  :: Blob
  pred       :: Array
  data_type  :: Type
  batch_size :: Int
  image_wh   :: NTuple{2, Int}
end

function preprocess{T <: ImageType}(classifier::ImageClassifier{T}, image::T)
  error("Not implemented")
end

function ImageClassifier{T}(::Type{T}, net::Net, softmax_blob_name::Symbol;
      channel_order = (1,2,3), # default RGB
      sp_order  = (1,2),       # default row-major (width,height,channels)
      grayscale = false,       # default not grayscale
      classes = [],            # default no name information available
    )

  if length(net.data_layers) != 1 || !isa(net.layers[net.data_layers[1]], MemoryDataLayer)
    error("The network should contain exactly one MemoryDataLayer")
  end
  data_layer = net.layers[net.data_layers[1]]
  batch_size = data_layer.batch_size
  data_type  = eltype(data_layer.data[1])
  image_wh   = (size(data_layer.data[1],1), size(data_layer.data[1],2))

  @assert length(data_layer.data) == 1
  if size(data_layer.data[1], 4) != batch_size
    # to make our life easier
    error("Please make the num dimension ($(size(data_layer.data[1],4))) of MemoryDataLayer's data source the same to batch_size ($(batch_size))")
  end

  pred_blob = net.output_blobs[softmax_blob_name]
  pred = Array(data_type, size(pred_blob))

  return ImageClassifier{T}(net, channel_order, sp_order, grayscale, classes,
      data_layer, pred_blob, pred, data_type, batch_size, image_wh)
end

################################################################################
# Interfaces
################################################################################
#-- Classify a single image via filename
function classify(classifier::ImageClassifier, filename::String)
  results = classify(classifier, String[filename])
  return results[1]
end

#-- Classify a bunch of images via filename
function classify{Tstr<:String,Timg<:ImageType}(classifier::ImageClassifier{Timg}, filenames::Vector{Tstr})
  return classify(classifier, map(name -> imread(Timg, name), filenames))
end

#-- Classify a single image
function classify(classifier::ImageClassifier, image::ImageType)
  results = classify(classifier, ImageType[image])
  return results[1]
end

#-- Classify a bunch of images
function classify(classifier::ImageClassifier, images::Vector{ImageType})
  results = classify_batch(classifier, images, 1)
  idx = classifier.batch_size+1
  while idx <= length(images)
    ret_batch = classify_batch(classifier, images, idx)
    results = [results, ret_batch]
    idx += length(res_batch)
  end

  return results
end

################################################################################
# Implementations
################################################################################
#-- Propress images
function preprocess(classifier::ImageClassifier, images::Vector{Image})
  map(img -> preprocess(classifier, img), images)
end

#-- Classify a batch or less
function classify_batch(classifier::ImageClassifier, images::Vector{Image}, idx::Int)
  #-- prepare the data
  idx_end = min(idx+classifier.batch_size-1, length(images))
  images = preprocess(classifier, images[idx:idx_end])
  for i = 1:length(images)
    classifier.data_layer.data[1][:,:,:,i] = images[i]
  end

  #-- run the network
  forward(classifier.net)
  copy!(classifier.pred, classifier.pred_blob)
  results = map(1:length(images)) do i
    pred = classifier.pred[:,:,:,i]
    ret = Array(Any, size(pred,1), size(pred,2))
    for w = 1:size(pred,1)
      for h = 1:size(pred,2)
        i_class = indmax(pred[w,h,:])
        if !isempty(classifier.classes)
          ret[w,h] = classifier.classes[i_class]
        else
          ret[w,h] = string(i_class-1) # 0-based class label
        end
      end
    end

    if size(pred,1) == 1 && size(pred,2) == 1
      ret = ret[1]
      pred = reshape(pred, size(pred,3), size(pred,4))
    end
    return (pred, ret)
  end

  return results
end
