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

  rc            :: RefCounter
end

# construct a new parameter
Parameter(name,blob,gradient,initializer,regularizer,constraint,lr) =
    Parameter(name, blob, gradient, initializer, regularizer, constraint, lr, RefCounter(1))

function make_parameter{N}(backend::Backend, name::String, data_type::Type, dims::NTuple{N,Int},
    init::Initializer, regu::Regularizer, cons::Constraint, lr::FloatingPoint)

  blob = make_blob(backend, data_type, dims)
  grad = make_blob(backend, data_type, dims)
  rc   = RefCounter(1)

  Parameter(name, blob, grad, init, regu, cons, lr, rc)
end

# make a shared parameter
function share_parameter(backend::Backend, param::Parameter)
  blob  = param.blob
  grad  = make_blob(backend, eltype(blob), size(blob))
  init  = NullInitializer()
  regu  = param.regularizer
  cons  = param.constraint
  lr    = param.learning_rate
  rc    = ref(param.rc)

  Parameter(param.name, blob, grad, init, regu, cons, lr, rc)
end

function destroy(param::Parameter)
  destroy(param.gradient)
  if dec(param.rc) == 0
    destroy(param.blob)
  end
end
