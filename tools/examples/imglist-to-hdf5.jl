# This example shows how to convert a file with a list of images
# to HDF5 format that Mocha can process
#
# Note this is a simple demo to show how to convert a typical
# dataset into HDF5 file that Mocha could handle. We do not
# perform any pre-processing on the images and simply assume
# all the images are of the same size and with the same
# number of channels.
#
# The format of the file is:
#
# ImageName,Label,TrainTest
# image1.jpg,009,Train
# image2.jpg,018,Train
# ....
# imageK.jpg,053,Test
# ...
# imageN.jpg,009,Test

using DataFrames
using Images
using Color
using HDF5
using Compat

############################################################
# Configuration
############################################################
image_list_file = "example.csv"
data_type       = Float32

############################################################
# Converting dataset
############################################################
datasets = Dict{AbstractString, HDF5File}()
dsets_idx = Dict{AbstractString, Int}()

df = readtable(image_list_file)
for i = 1:size(df,1)
  image_fn = df[i, :ImageName]
  label    = df[i, :Label]
  dset_key = df[i, :TrainTest]

  println(image_fn)

  image = separate(convert(Image{RGB}, imread(image_fn)))
  data  = convert(Array, image)

  if !haskey(datasets, dset_key)
    #---------------
    # create dataset
    #---------------

    # count how many elements in this dataset
    dataset_count = sum(df[:TrainTest] .== dset_key)

    h5 = h5open("$dset_key.hdf5", "w")
    dset_data = d_create(h5, "data", datatype(data_type),
        dataspace(size(data)..., dataset_count))
    dset_label = d_create(h5, "label", datatype(data_type),
        dataspace(1, dataset_count))
    datasets[dset_key] = h5
    dsets_idx[dset_key] = 0
  end

  h5 = datasets[dset_key]
  idx = dsets_idx[dset_key]
  idx += 1
  h5["label"][1,idx] = label
  h5["data"][map(x -> 1:x, size(data))..., idx] = data
end

for h5 in values(datasets)
  close(h5)
end
