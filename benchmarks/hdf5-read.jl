#################################################################################
# Test on my laptop
# | Row | Function              | Average  | Relative | Replications |
# |-----|-----------------------|----------|----------|--------------|
# | 1   | "batch_read"          | 0.246701 | 1.0      | 5            |
# | 2   | "batch_read_mmap"     | 0.304279 | 1.23339  | 5            |
# | 3   | "batch_read_blascopy" | 0.385944 | 1.56442  | 5            |
# | 4   | "rand_read"           | 3.49723  | 14.176   | 5            |
# | 5   | "rand_read_mmap"      | 0.386288 | 1.56582  | 5            |
#
# Test on openmind server
# | Row | Function              | Average  | Relative | Replications |
# |-----|-----------------------|----------|----------|--------------|
# | 1   | "batch_read"          | 0.190785 | 1.0      | 5            |
# | 2   | "batch_read_mmap"     | 0.372572 | 1.95284  | 5            |
# | 3   | "batch_read_blascopy" | 0.521257 | 2.73217  | 5            |
# | 4   | "rand_read"           | 3.53322  | 18.5194  | 5            |
# | 5   | "rand_read_mmap"      | 0.421925 | 2.21152  | 5            |
#################################################################################
using HDF5
println("HDF5 module loaded")

path = joinpath(dirname(@__FILE__), "..", "examples", "mnist", "data", "train.hdf5")
println("Testing with HDF5 file $path")

batch_size = 100
n_batch = 600

function batch_read(mmap::Bool, blas::Bool)
  println("batch_read $(mmap ? "(mmap)" : "")...")
  h5open(path) do file
    dset = file["data"]
    if mmap
      dset = readmmap(dset)
    end
    data = zeros(size(dset,1),size(dset,2),size(dset,3),batch_size)
    @assert batch_size*n_batch == size(dset, 4)
    if blas
      @assert mmap

      data_type = eltype(dset)
      len = length(data)
      len_byte = len * sizeof(data_type)

      for i = 1:n_batch
        BLAS.blascopy!(len, convert(Ptr{data_type},dset) + (i-1)*len_byte, 1,
            convert(Ptr{data_type},data), 1)
      end
    else
      for i = 1:n_batch
        data = dset[:,:,:,(i-1)*batch_size+1:i*batch_size]
      end
    end
  end
end

function rand_read(mmap::Bool)
  println("rand_read $(mmap ? "(mmap)" : "")...")
  rp = randperm(n_batch*batch_size)
  h5open(path) do file
    dset = file["data"]
    if mmap
      dset = readmmap(dset)
    end
    data = zeros(size(dset,1),size(dset,2),size(dset,3),batch_size)
    @assert batch_size*n_batch == size(dset, 4)
    if mmap
      for i = 1:n_batch
        data = dset[:,:,:,rp[(i-1)*batch_size+1:i*batch_size]]
      end
    else
      for i = 1:n_batch
        for j = 1:batch_size
          data[:,:,:,j] = dset[:,:,:,rp[(i-1)*batch_size+j]]
        end
      end
    end
  end
end

batch_read() = batch_read(false,false)
batch_read_mmap() = batch_read(true,false)
batch_read_blascopy() = batch_read(true,true)
rand_read() = rand_read(false)
rand_read_mmap() = rand_read(true)

using Benchmark
df = compare([batch_read, batch_read_mmap, batch_read_blascopy, rand_read, rand_read_mmap], 5)
println("$df")
