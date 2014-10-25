using HDF5

############################################################
# Prepare Data for Testing
############################################################
n_batch = 5
data_dim = (2,3,4)

data_all = [int(1000*rand(x*n_batch, data_dim...)) for x in [1 2 1]]
h5fn_all = [tempname() + ".hdf5" for x in 1:length(data_all)]

source_fn = tempname() + ".txt"
open(source_fn, "w") do s
  for fn in h5fn_all
    println(s, fn)
  end
end

for i = 1:length(data_all)
  h5 = h5open(h5fn_all[i], "w")
  h5["data"] = data_all[i]
  close(h5)
end

############################################################
# Setup
############################################################

# batch size is determined by 
layer = HDF5DataLayer(; source = source_fn, tops = String["data"])
state = setup(layer)

