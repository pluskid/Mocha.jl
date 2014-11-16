export PoolingFunction, Pooling
abstract PoolingFunction
abstract StdPoolingFunction <: PoolingFunction # built-in poolings

module Pooling
using ..Mocha.StdPoolingFunction

type Max <: StdPoolingFunction end
type Mean <: StdPoolingFunction end

end # module Pooling
