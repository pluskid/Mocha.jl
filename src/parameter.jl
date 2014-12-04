export Parameter

type Parameter
  name          :: String
  blob          :: Blob
  gradient      :: Blob
  initializer   :: Initializer
  regularizer   :: Regularizer
  constraint    :: Constraint
  learning_rate :: FloatingPoint # relative learning rate
  owner         :: Bool # if owner of the parameter blob
end

# construct a new parameter
Parameter(name,blob,gradient,initializer,regularizer,constraint,lr) =
    Parameter(name, blob, gradient, initializer, regularizer, constraint, lr, true)

# make a shared parameter
function make_shared_parameter(backend::Backend, param::Parameter)
  Parameter(param.name, param.blob,
      make_blob(backend, eltype(param.gradient), size(param.gradient)),
      NullInitializer(), param.regularizer, param.constraint,
      param.learning_rate, false)
end

function destroy(param::Parameter)
  destroy(param.gradient)
  if param.owner
    destroy(param.blob)
  end
end
