export Parameter

type Parameter
  blob        :: Blob
  gradient    :: Blob
  initializer :: Initializer
  #regularizer
  #weight_decay
  #learning_rate
end
