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
include("layers/channel-pooling.jl")

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


