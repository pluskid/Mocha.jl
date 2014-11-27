using Images # requires Package Images.jl
using Color
using Mocha

type ImageClassifier
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

function ImageClassifier(net::Net, softmax_blob_name::Symbol;
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

  return ImageClassifier(net, channel_order, sp_order, grayscale, classes,
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
function classify{T<:String}(classifier::ImageClassifier, filenames::Vector{T})
  return classify(classifier, map(Images.imread, filenames))
end

#-- Classify a single image
function classify(classifier::ImageClassifier, image::Image)
  results = classify(classifier, Image[image])
  return results[1]
end

#-- Classify a bunch of images
function classify(classifier::ImageClassifier, images::Vector{Image})
  results = classify_batch(classifier, images, 1)
  idx = classifier.batch_size+1
  while idx <= length(images)
    res_batch = classify_batch(classifier, images, idx)
    results = [results, res_batch]
    idx += length(res_batch)
  end

  return results
end

################################################################################
# Implementations
################################################################################
#-- Propress images
function preprocess(classifier::ImageClassifier, images::Vector{Image})
  map(images) do image
    if (width(img), height(img)) != classifier.image_wh
      error("Image size ($(width(img)),$(height(img))) does not match the network input $(classifier.image_wh), and Julia does not have imresize yet........")
    end

    if classifier.grayscale
      if colorspace(image) != "Gray"
        image = convert(Image{Gray}, image)
      end
    else
      image = convert(Image{RGB}, image)
      image = separate(image) # separate color channels
    end

    data = convert(Array, image)
    data = reshape(data, size(data,1),size(data,2),size(data,3))
    if spatialorder(image) == ["x","y"]
      # row major
    elseif spatialorder(image) == ["y","x"]
      data = permutedims(data, (2,1,3)) # permute to row-major
    else
      error("Unknown spatialorder: $(spatialorder(image))")
    end

    # now data is in canonical row-major RGB/Gray format
    if classifier.sp_order != (1,2)
      @assert classifier.sp_order == (2,1)
      data = permutedims(data, (2,1,3))
    end

    if classifier.grayscale
      data2 = convert(Array{classifier.data_type}, data)
    else
      data2 = Array(classifier.data_type, tuple(classifier.image_wh..., 3))
      for i = 1:3
        data2[:,:,i] = convert(Array{classifier.data_type},
            data[:,:,classifier.channel_order[i]])
      end
    end

    data2 # pre-processed raw image data
  end
end

#-- Classify a batch or less
function classify_batch(classifier::ImageClassifier, images::Vector{Image}, idx::Int)
  #-- prepare the data
  idx_end = min(idx+classifier.batch_size-1, length(images))
  images = preprocess(classifier, images[idx:idx_end])
  for i = 1:length(images)
    # we scale by 255 because it seems that Images.jl automatically map
    # [0,256) to [0,1), but the mean_file we converted from Caffe assumes
    # the data range is [0, 256)
    classifier.data_layer.data[1][:,:,:,i] = 255*images[i]
  end

  #-- run the network
  forward(classifier.net)
  copy!(classifier.pred, classifier.pred_blob)
  results = map(1:length(images)) do i
    pred = classifier.pred[:,:,:,i]
    if !isempty(classifier.classes)
      ret = Array(Any, size(pred,1), size(pred,2))
      for w = 1:size(pred,1)
        for h = 1:size(pred,2)
          ret[w,h] = classifier.classes[indmax(pred[w,h,:])]
        end
      end
    else
      ret = pred
    end

    if size(pred,1) == 1 && size(pred,2) == 1
      ret = reshape(ret, length(ret))
    end

    ret
  end

  return results
end
