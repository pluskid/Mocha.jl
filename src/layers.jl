export Layer, LayerState
export DataLayer, LossLayer, StatLayer, CompLayer

export HDF5DataLayer
export InnerProductLayer
export SquareLossLayer

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
#
# If the layer needs back propagated gradient from upper
# layer, then the following fields are required:
#
# - blobs_diff: An array of blobs. Corresponding to the
#       gradient of the objective function with respect
#       to the output of this layer in the backward pass.
#       Note this value is computed by the upper layer.
#
# If the layer has its own parameters that need to be updated
# during optimization, the following fields should be defined
# 
# - parameters: vector of Blob.
# - gradients: vector of Blob, gradients of parameters, 
#       should be computed in the backward pass.
#
# Then the following functions need to be defined
#
# - setup(layer :: MyLayer, inputs :: Vector{Blob})
#   This function construct the layer state object and do
#   necessary initialization. The inputs are initialized
#   with proper shape, but not necessarily with valid data
#   values. The constructed layer state should be returned.
#
# - forward(state :: MyLayerState, inputs :: Vector{Blob})
#   This function do the forward computation: inputs are
#   forward computed output from bottom layers.
#
# - backward(state::MyLayerState,inputs::Vector{Blob},diffs::Vector{Blob})
#   This function do the backward computation: inputs are
#   the same as in forward, diffs contains blobs to hold
#   gradient with respect to the bottom layer input. Some
#   blob in this vector might be "undefined", meaning that
#   blob do not want to get back propagated gradient. This
#   procedure also compute gradient with respect to layer
#   parameters if any.
#
# - update(state :: MyLayerState)
############################################################

abstract Layer      # define layer type, parameters
abstract LayerState # hold layer state, filters

abstract DataLayer <: Layer # Layer that provide data
abstract LossLayer <: Layer # Layer that defines loss function for learning
abstract StatLayer <: Layer # Layer that provide statistics (e.g. Accuracy)
abstract CompLayer <: Layer # Layer that do computation

#############################################################
# Data Layers
#############################################################
include("layers/hdf5-data.jl")

#############################################################
# Loss Layers
#############################################################
include("layers/square-loss.jl")

#############################################################
# General Computation Layers
#############################################################
include("layers/inner-product.jl")

