# convert binary into HDF5 data

using HDF5

datasets = [("train", ["data_batch_$i.bin" for i in 1:5]),
            ("test", ["test_batch.bin"])]

const width      = 32
const height     = 32
const channels   = 3
const batch_size = 10000

mean_model = zeros(Float32, width, height, channels, 1)

for (key, sources) in datasets
  h5open("$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), 
        dataspace(width, height, channels, batch_size * length(sources)))
    dset_label = d_create(h5, "label", datatype(Float32), 
        dataspace(1, batch_size * length(sources)))

    for n = 1:length(sources)
      open("cifar-10-batches-bin/$(sources[n])") do f
        println("Processing $(sources[n])...")
        mat = readbytes(f, (1 + width*height*channels) * batch_size)
        mat = reshape(mat, 1+width*height*channels, batch_size)

        # random shuffle within batch
        rp = randperm(batch_size)

        label = convert(Array{Float32},mat[1, rp])
        # If I divide by 256 as in the MNIST example, then
        # training on the giving DNN gives me random
        # performance: objective function not changing,
        # and test performance is always 10%...
        # The same results could be observed when
        # running Caffe, as our HDF5 dataset is
        # compatible with Caffe.
        img = convert(Array{Float32},mat[2:end, rp])
        img = reshape(img, width, height, channels, batch_size)

        if key == "train"
          # only accumulate mean from the training data
          global mean_model
          mean_model = (batch_size*mean_model + sum(img, 4)) / (n*batch_size)
        end

        index = (n-1)*batch_size+1:n*batch_size
        dset_data[:,:,:,index] = img
        dset_label[:,index] = label
      end
    end

    # but apply mean subtraction for both training and testing data
    println("Subtracting the mean...")
    for n = 1:length(sources)
      index = (n-1)*batch_size+1:n*batch_size
      dset_data[:,:,:,index] = dset_data[:,:,:,index] .- mean_model
    end
  end
end

