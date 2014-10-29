abstract PoolingFunction

module Pooling
using ..Mocha.PoolingFunction

type Max <: PoolingFunction
  masks # mask used to indicate which position is max-picked
  Max() = new() # masks will be initialized by whoever use this
end
type Mean <: PoolingFunction end

end # module Pooling
