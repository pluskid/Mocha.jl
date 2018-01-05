export LayerState

export HDF5DataLayer, MemoryDataLayer, AsyncHDF5DataLayer
export InnerProductLayer, ConvolutionLayer, PoolingLayer, SoftmaxLayer
export PowerLayer, SplitLayer, ElementWiseLayer, ChannelPoolingLayer
export LRNLayer, DropoutLayer, ReshapeLayer, ArgmaxLayer, HDF5OutputLayer
export CropLayer, ConcatLayer, RandomMaskLayer, TiedInnerProductLayer
export IdentityLayer, Index2OnehotLayer, MemoryOutputLayer
export SquareLossLayer, SoftmaxLossLayer, MultinomialLogisticLossLayer
export SoftlabelSoftmaxLossLayer, WassersteinLossLayer, HingeLossLayer
export AccuracyLayer, BinaryAccuracyLayer, BinaryCrossEntropyLossLayer

export RandomNormalLayer
export GaussianKLLossLayer
export setup, forward, backward, shutdown

export reset_statistics
export freeze!, unfreeze!, is_frozen

############################################################
# Implementing a Layer
#
# A Layer object is a configuration of how a layer is going
# to behave. The following fields are needed by the neural
# network engine:
#
# - tops: An array of strings, as the name of output blobs.
#       Note tops is optional for LossLayer and StatLayer.
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
# Optional each layer can have an activation function
# by defining a field called neuron. The activation
# function computation (both forward and backward)
# are handled automatically by the engine.
#
# If the layer has its own parameters that need to be updated
# during optimization, the following field should be defined
#
# - parameters: vector of Parameters
#
# Note the layer needs to compute the gradient of the parameter
# during the backward pass, but everything else will be taken
# care of automatically by the engine, including the followings:
#
# - initialize the parameters before optimization starts
# - compute the regularization and its gradient during forward
#   and backward pass, respectively
# - update the parameter after one forward-backward pass
#
# Then the following functions need to be defined
#
# - setup(backend::Backend,layer::MyLayer,inputs::Vector{Blob})
#   This function construct the layer state object and do
#   necessary initialization. The inputs are initialized
#   with proper shape, but not necessarily with valid data
#   values. The constructed layer state should be returned.
#
# - forward(backend::Backend,state::MyLayerState,inputs::Vector{Blob})
#   This function do the forward computation: inputs are
#   forward computed output from bottom layers.
#
# - backward(backend::Backend,state::MyLayerState,inputs::Vector{Blob},diffs::Vector{Blob})
#   This function do the backward computation: inputs are
#   the same as in forward, diffs contains blobs to hold
#   gradient with respect to the bottom layer input. Some
#   blob in this vector might be "undefined", meaning that
#   blob do not want to get back propagated gradient. This
#   procedure also compute gradient with respect to layer
#   parameters if any.
############################################################


############################################################
# Default layer characterizations
############################################################
@characterize_layer(Layer,
  is_source  => false, # data layer, takes no bottom blobs
  is_sink    => false, # top layer, produces no top blobs (loss, accuracy, etc.)
  has_param  => false, # contains trainable parameters
  has_neuron => false, # has a neuron
  can_do_bp  => false, # can do back-propagation
  is_inplace => false, # do inplace computation, does not has own top blobs
  has_loss   => false, # produce a loss
  has_stats  => false, # produce statistics
)

function setup(backend::Backend, layer::Layer, inputs::Vector{Blob}, diffs::Vector{Blob})
  error("Not implemented, should setup layer state")
end

# Default overload when there is no shared_parameters
function setup(backend::Backend, layer::Layer, shared_parameters, inputs::Vector{Blob}, diffs::Vector{Blob})
  setup(backend, layer, inputs, diffs)
end

function shutdown(backend::Backend, state::LayerState)
  error("Not implemented, please define an empty function explicitly if not needed")
end

function forward(backend::Backend, state::LayerState, inputs::Vector{Blob})
  error("Not implemented, should do forward computing")
end

function backward(backend::Backend, state::LayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  error("Not implemented, please define an empty function explicitly if not needed")
end

function param_key(layer::Layer)
  @assert has_param(layer)
  key = layer.param_key
  return isempty(key) ? layer.name : key
end

# If a layer state is frozen, its parameters will not be trained, also there is no
# need to compute the gradients for the parameters
function freeze!(state::LayerState)
  @assert !has_param(state.layer) "Layers with parameters should implement their own freeze function"
  # freeze has no effects for layers without parameters
end
function unfreeze!(state::LayerState)
  @assert !has_param(state.layer) "Layers with parameters should implement their own unfreeze function"
end
function is_frozen(state::LayerState)
  @assert !has_param(state.layer) "Layers with parameters should implement their own is_frozen function"
  false # layers without parameters are never frozen
end

#############################################################
# Display layers
#############################################################
import Base.show
export show, show_layer

function show(io::IO, layer::Layer)
  print(io, "$(typeof(layer))($(layer.name))")
end
function show_layer_details(io::IO, state::LayerState)
  # overload this function to add layer details
end
function show_layer(io::IO, state::LayerState, inputs::Vector{Blob})
  println(io, "............................................................")
  layer_title = " *** $(state.layer)"
  if isa(io, Base.TTY)
    print_with_color(:blue, io, layer_title)
  else
    print(io, layer_title)
  end
  println(io)

  show_layer_details(io, state)
  if !is_source(state.layer)
    println(io, "    Inputs ----------------------------")
    for i = 1:length(inputs)
      println(io, "     $(@sprintf("%9s", state.layer.bottoms[i])): $(inputs[i])")
    end
  end
  if !is_sink(state.layer) && !is_inplace(state.layer)
    println(io, "    Outputs ---------------------------")
    for i = 1:length(state.blobs)
      println(io, "     $(@sprintf("%9s", state.layer.tops[i])): $(state.blobs[i])")
    end
  end
end

#############################################################
# Data Layers
#############################################################
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")
include("layers/async-hdf5-data.jl")

#############################################################
# General Computation Layers
#############################################################
include("layers/inner-product.jl")
include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/softmax.jl")
include("layers/power.jl")
include("layers/split.jl")
include("layers/element-wise.jl")
include("layers/channel-pooling.jl")
include("layers/lrn.jl")
include("layers/dropout.jl")
include("layers/reshape.jl")
include("layers/crop.jl")
include("layers/concat.jl")
include("layers/random-mask.jl")
include("layers/tied-inner-product.jl")
include("layers/identity.jl")
include("layers/random-normal.jl")

#############################################################
# Utility layers
#############################################################
include("layers/argmax.jl")
include("layers/hdf5-output.jl")
include("layers/index2onehot.jl")
include("layers/memory-output.jl")

#############################################################
# Loss Layers
#############################################################
include("layers/square-loss.jl")
include("layers/multinomial-logistic-loss.jl")
include("layers/softmax-loss.jl")
include("layers/softlabel-softmax-loss.jl")
include("layers/wasserstein-loss.jl")
include("layers/binary-cross-entropy-loss.jl")
include("layers/gaussian-kl-loss.jl")
include("layers/hinge-loss.jl")

#############################################################
# Statistics Layers
#############################################################
include("layers/accuracy.jl")
include("layers/binary-accuracy.jl")

