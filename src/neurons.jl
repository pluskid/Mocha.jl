export ActivationFunction
export Neurons

export forward, backward

############################################################
# Each activation function / neuron should be a sub type
# of ActivationFunction and should implement the following
# methods.
#
# - forward(sys :: System, neuron :: MyActivationFunction, input :: Blob)
#   Update the input blob by the activation function. Computation should be done in-place.
#
# - backward(sys :: System, neuron :: MyActivationFunction, input :: Blob, gradient :: Blob)
#   The gradient blob contains the gradient with respect to the output of the activation
#   function, update (IN-PLACE) the gradient to be with respect to the input of the activation
#   function.
############################################################

abstract ActivationFunction

############################################################
# A module to hold built-in activation functions to avoid
# messy namespace
############################################################
module Neurons
using ..Mocha.ActivationFunction
# Identity 
type Identity <: ActivationFunction
end

# Rectified-Linear 
type ReLU <: ActivationFunction
end
end # module Neurons

############################################################
# Identity
############################################################
function forward(sys :: System{CPU}, neuron :: Neurons.Identity, input :: Blob)
  # do nothing
end
function backward(sys :: System{CPU}, neuron :: Neurons.Identity, input :: Blob, gradient :: Blob)
  # do nothing
end

############################################################
# Rectified-Linear 
############################################################
function forward(sys :: System{CPU}, neuron :: Neurons.ReLU, input :: Blob)
  input.data[:] = max(input.data[:], 0)
end
function backward(sys :: System{CPU}, neuron :: Neurons.ReLU, input :: Blob, gradient :: Blob)
  gradient.data[:] .*= (input.data[:] .>= 0)
end
