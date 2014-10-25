export Layer, LayerState
export DataLayer, CostLayer, StatLayer, CompLayer

export HDF5DataLayer
export InnerProductLayer

export setup, forward

abstract Layer      # define layer type, parameters
abstract LayerState # hold layer state, filters

abstract DataLayer <: Layer # Layer that provide data
abstract CostLayer <: Layer # Layer that defines cost function for learning
abstract StatLayer <: Layer # Layer that provide statistics (e.g. Accuracy)
abstract CompLayer <: Layer # Layer that do computation

#############################################################
# Data Layers
#############################################################
include("layers/hdf5-data.jl")


#############################################################
# General Computation Layers
#############################################################
include("layers/inner-product.jl")
