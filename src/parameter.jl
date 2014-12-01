export Parameter

type Parameter
  name          :: String
  blob          :: Blob
  gradient      :: Blob
  initializer   :: Initializer
  regularizer   :: Regularizer
  constraint    :: Constraint
  learning_rate :: FloatingPoint # relative learning rate
end
