# An Initializer is used to initialize network parameters
export Initializer
export ConstantInitializer

abstract Initializer # The root type of all initializer

type ConstantInitializer <: Initializer
  value
end

function init(initializer::ConstantInitializer, blob::Blob)
  blob[:] = initializer.value
end

