export Parameter

type Parameter
  blob        :: Blob
  gradient    :: Blob
  initializer :: Initializer
  regularizer :: Regularizer
  #learning_rate
end
