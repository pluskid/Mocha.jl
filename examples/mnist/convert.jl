# convert binary into HDF5 data

using HDF5
using Compat

srand(12345678)

datasets = @compat(Dict("train" => ["train-labels-idx1-ubyte","train-images-idx3-ubyte"],
            "test" => ["t10k-labels-idx1-ubyte","t10k-images-idx3-ubyte"]))

for key in keys(datasets)
  label_fn, data_fn = datasets[key]
  label_f = open(label_fn)
  data_f  = open(data_fn)

  label_header = read(label_f, Int32, 2)
  @assert ntoh(label_header[1]) == 2049
  n_label = @compat(Int(ntoh(label_header[2])))
  data_header = read(data_f, Int32, 4)
  @assert ntoh(data_header[1]) == 2051
  n_data = @compat(Int(ntoh(data_header[2])))
  @assert n_label == n_data
  h = @compat(Int(ntoh(data_header[3])))
  w = @compat(Int(ntoh(data_header[4])))

  n_batch = 1
  @assert n_data % n_batch == 0
  batch_size = @compat(Int(n_data / n_batch))

  println("Exporting $n_data digits of size $h x $w")

  h5open("$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))

    for i = 1:n_batch
      idx = (i-1)*batch_size+1:i*batch_size
      println("  $idx...")

      idx = collect(idx)
      rp = randperm(length(idx))

      img = readbytes(data_f, batch_size * h*w)
      img = convert(Array{Float32},img) / 256 # scale into [0,1)
      class = readbytes(label_f, batch_size)
      class = convert(Array{Float32},class)

      for j = 1:length(idx)
        r_idx = rp[j]
        dset_data[:,:,1,idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
        dset_label[1,idx[j]] = class[r_idx]
      end
    end
  end

  close(label_f)
  close(data_f)
end

