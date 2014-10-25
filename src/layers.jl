abstract Layer

abstract DataLayer <: Layer # Layer that provide data
abstract CostLayer <: Layer # Layer that defines cost function for learning
abstract StatLayer <: Layer # Layer that provide statistics (e.g. Accuracy)
abstract CompLayer <: Layer # Layer that do computation

@defstruct HDF5DataLayer DataLayer (
  (source :: String = "", source != ""),
  batch_size :: Unsigned = 0, # 0 means varied batch_size, all samples from a file
  tops :: Vector{String} = String["data","label"]
)
