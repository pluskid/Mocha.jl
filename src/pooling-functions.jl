export PoolingFunction, Pooling
@compat abstract type PoolingFunction end
@compat abstract type StdPoolingFunction <: PoolingFunction end # built-in poolings

module Pooling
using ..Mocha.StdPoolingFunction

type Max <: StdPoolingFunction end
type Mean <: StdPoolingFunction end

end # module Pooling
