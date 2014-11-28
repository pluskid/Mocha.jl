################################################################################
# A simple image classifier wrapper of Mocha, based on the Images.jl package.
# However, Images.jl does NOT support resizing images yet... so if you need
# to classify images that is not of the same size of images you used to train
# your model, consider using pycall-image-classifier.jl instead.
################################################################################

include(joinpath(dirname(@__FILE__), "base-image-classifier.jl"))

using Color
using Images

type JuliaImage <: ImageType
  img :: Image
end

function ImageClassifier(net::Net, softmax_blob_name::Symbol; kw...)
  ImageClassifier(JuliaImage, net, softmax_blob_name; kw...)
end

function imread(::Type{JuliaImage}, filename::String)
  JuliaImage(imread(filename))
end

function preprocess(classifier::ImageClassifier{JuliaImage}, image::JuliaImage)
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

