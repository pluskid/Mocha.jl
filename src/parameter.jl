export Parameter
export make_parameter, share_parameter

type Parameter <: AbstractParameter
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

function make_parameter(backend::Backend, name::String, data_type::Type, dims::NTuple{4,Int},
    init::Initializer, regu::Regularizer, cons::Constraint, lr::FloatingPoint)

  blob = make_blob(backend, data_type, dims)
  grad = make_blob(backend, data_type, dims)
  owner = true

  Parameter(name, blob, grad, init, regu, cons, lr, owner)
end

# make a shared parameter
function share_parameter(backend::Backend, param::Parameter)
  blob  = param.blob
  grad  = make_blob(backend, eltype(blob), size(blob))
  init  = NullInitializer()
  regu  = param.regularizer
  cons  = param.constraint
  lr    = param.learning_rate
  owner = false

  Parameter(param.name, blob, grad, init, regu, cons, lr, owner)
end

function destroy(param::Parameter)
  destroy(param.gradient)
  if param.owner
    destroy(param.blob)
  end
end
