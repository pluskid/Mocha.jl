# convert binary into HDF5 data
# 
# This modification to the original convert scrips adds a dataset to
# the HDF5 file that has data in the same order as the test set, but
# does not include the mean scaling and preserves the pixel data in
# the range [0.0:1.0] necessary to display it with the ImageView
# package.
#
# This is completed by one of two methods...
#   1. Copying the test set before applying the mean shift
#   2. Reconstructing the orignal data be reversing the mean shift
#      after all other operations are completed.  I think I like this
#      version better since it seems more sequential and the numerical
#      accuracy is important since the purpose of the visual set is
#      simply to "see" the images and compare them against the 
#      predicted labels.

using HDF5

const width      = 32
const height     = 32
const channels   = 3
const batch_size = 10000
const data_filedir = "cifar_data"

# random shuffle permutation within batch
rp = randperm(batch_size)

# mean model will be used to create sets with zero mean
mean_model = zeros(Float32, width, height, channels, 1)

#create the training set
datasets = [("train", ["data_batch_$i.bin" for i in 1:5])]

for (key, sources) in datasets
  h5open("$data_filedir/$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), 
        dataspace(width, height, channels, batch_size * length(sources)))
    dset_label = d_create(h5, "label", datatype(Float32), 
        dataspace(1, batch_size * length(sources)))

    for n = 1:length(sources)
      open("$data_filedir/cifar-10-batches-bin/$(sources[n])") do f
        println("Processing $(sources[n])...")

        mat = readbytes(f, (1 + width*height*channels) * batch_size)
        mat = reshape(mat, 1+width*height*channels, batch_size)

        label = convert(Array{Float32},mat[1, rp])
        
        # Note that we read one byte per pixel channel value.  These bytes
        # represent integer values in range [0:255].
        img = convert(Array{Float32},mat[2:end, rp])
        img = reshape(img, width, height, channels, batch_size)

        # accumulate mean from the training data
        # TODO - Take a closer look at the way the mean model is being 
        #        computed.  Not sure why we are dividing by n*batch_size.
        global mean_model
        mean_model = (batch_size*mean_model + sum(img, 4)) / (n*batch_size)

        index = (n-1)*batch_size+1:n*batch_size
        dset_data[:,:,:,index] = img
        dset_label[:,index] = label
      end
    end

    # apply mean subtraction for training data
    println("Subtracting the mean...")
    for n = 1:length(sources)
      index = (n-1)*batch_size+1:n*batch_size
      dset_data[:,:,:,index] = dset_data[:,:,:,index] .- mean_model
    end
  end   #Close hdf5 file
end   #Close iteration

#Create the testing and visual datasets
println("Creating the visual and test datasets...")
vis_fid = h5open("$data_filedir/visual.hdf5", "w")
vis_data = d_create(vis_fid, "data", datatype(Float32), 
    dataspace(width, height, channels, batch_size))
vis_label = d_create(vis_fid, "label", datatype(Float32), 
    dataspace(1, batch_size)) 

test_fid = h5open("$data_filedir/test.hdf5", "w")
test_data = d_create(test_fid, "data", datatype(Float32), 
    dataspace(width, height, channels, batch_size))
test_label = d_create(test_fid, "label", datatype(Float32), 
    dataspace(1, batch_size)) 

open("$data_filedir/cifar-10-batches-bin/test_batch.bin") do f
    println("Processing test batch...")

    mat = readbytes(f, (1 + width*height*channels) * batch_size)
    mat = reshape(mat, 1+width*height*channels, batch_size)

    label = convert(Array{Float32},mat[1, rp])

    # Note that we read one byte per pixel channel value.  These bytes
    # represent integer values in range [0:255].
    img = convert(Array{Float32},mat[2:end, rp])
    img = reshape(img, width, height, channels, batch_size)

    index = 1:batch_size
    test_data[:,:,:,index] = img
    test_label[:,index] = label
    vis_data[:,:,:,index] = img
    vis_label[:,index] = label
end
    
# apply mean subtraction for test data
println("Subtracting the mean from the test_data...")
index = 1:batch_size
test_data[:,:,:,index] = test_data[:,:,:,index] .- mean_model

# clamp the variance in the visual data to the range [0.0:1.0]
println("Shifting variance for visual data to the range [0.0:1.0]...")
index = 1:batch_size
vis_data[:,:,:,index] = vis_data[:,:,:,index] ./ 255.0

#close the datasets
close(vis_fid)
close(test_fid)
