#############################################################
# Data Layers
#############################################################
include("layers/hdf5-data.jl")

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
include("layers/dropout.jl")
include("layers/argmax.jl")
include("layers/crop.jl")
include("layers/concat.jl")
include("layers/tied-inner-product.jl")

#############################################################
# Loss Layers
#############################################################
include("layers/square-loss.jl")
include("layers/multinomial-logistic-loss.jl")
include("layers/softmax-loss.jl")

#############################################################
# Statistics Layers
#############################################################
include("layers/accuracy.jl")


