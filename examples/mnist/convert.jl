# convert binary into HDF5 data

using HDF5

datasets = ["train" => ["train-labels-idx1-ubyte","train-images-idx3-ubyte"],
            "test" => ["t10k-labels-idx1-ubyte","t10k-images-idx3-ubyte"]]

for key in keys(datasets)
  label_fn, data_fn = datasets[key]
  label_f = open(label_fn)
  data_f  = open(data_fn)

  label_header = read(label_f, Int32, 2)
  @assert ntoh(label_header[1]) == 2049
  n_label = int(ntoh(label_header[2]))
  data_header = read(data_f, Int32, 4)
  @assert ntoh(data_header[1]) == 2051
  n_data = int(ntoh(data_header[2]))
  @assert n_label == n_data
  h = int(ntoh(data_header[3]))
  w = int(ntoh(data_header[4]))

  n_batch = 10
  @assert n_data % n_batch == 0
  batch_size = int(n_data / n_batch)

  println("Exporting $n_data digits of size $h x $w")

  h5open("$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(n_data,1,h,w))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(n_data,1,1,1))

    for i = 1:n_batch
      idx = (i-1)*batch_size+1:i*batch_size
      println("  $idx...")

      img = readbytes(data_f, batch_size * h*w)
      img = convert(Array{Float32},img) / 256 # scale into [0,1)
      dset_data[idx,1,:,:] = img

      class = readbytes(label_f, batch_size)
      class = convert(Array{Float32},class) + 1 # map to class 1..10 instead of 0..9
      dset_label[idx,:,1,1] = class
    end
  end

  close(label_f)
  close(data_f)
end

