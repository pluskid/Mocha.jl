using HDF5

import Base.eltype
export eltype

# a function copied from HDF5.jl master, should remove
# this after HDF5.jl released a new version with this
# function shipped
function eltype(dset::Union(HDF5.HDF5Dataset, HDF5.HDF5Attribute))
    T = Any
    dtype = HDF5.datatype(dset)
    try
        T = HDF5.hdf5_to_julia_eltype(dtype)
    finally
        HDF5.close(dtype)
    end
    T
end
