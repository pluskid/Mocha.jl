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
@defstruct HDF5DataLayer DataLayer (
  (source :: String = "", source != ""),
  (batch_size :: Int = 0, batch_size > 0), 
  tops :: Vector{String} = String["data","label"]
)

#############################################################
# General Computation Layers
#############################################################
@defstruct InnerProductLayer CompLayer (
  (output_dim :: Int = 0, output_dim > 0),
  (tops :: Vector{String} = String[], length(tops) == 1)
)
