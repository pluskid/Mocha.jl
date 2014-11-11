export PoolingFunction, Pooling
abstract PoolingFunction

module Pooling
using ..Mocha.PoolingFunction

type Max <: PoolingFunction end
type Mean <: PoolingFunction end

end # module Pooling
