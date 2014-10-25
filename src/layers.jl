export Layer, LayerState
export DataLayer, CostLayer, StatLayer, CompLayer

export HDF5DataLayer
export InnerProductLayer

export setup, forward

############################################################
# Implementing a Layer
#
# A Layer object is a configuration of how a layer is going
# to behave. The following fields are needed by the neural
# network engine:
# 
# - tops: An array of strings, as the name of output blobs
# - bottoms: An array of strings for the name of input blobs.
#       Data Layers can omit this field.
#
# To add functional operation for the layer, a layer state
# type should be defined as a sub-type of LayerState. A
# LayerState object stores all the states and intermediate
# computation results of each layer. The following fields
# are needed by the neural network engine:
#
# - blobs: An array of blobs. Corresponding to the output
#       of this layer in the forward pass.
# - blobs_diff: An array of blobs. Corresponding to the
#       gradient of the objective function with respect
#       to the output of this layer in the backward pass.
#       Note this value is computed by the upper layer.
#       If the layer does not need back propagation, then
#       this field could be omitted.
#
# Then the following functions need to be defined
#
# - setup(layer :: MyLayer, inputs :: Vector{Blob})
#   This function construct the layer state object and do
#   necessary initialization. The inputs are initialized
#   with proper shape, but not necessarily with valid data
#   values. The constructed layer state should be returned.
#
# - forward(state :: MayLayerState, inputs :: Vector{Blob})
#   This function do the forward computation 
############################################################

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
