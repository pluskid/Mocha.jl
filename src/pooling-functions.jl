export PoolingFunction, Pooling
@compat abstract type PoolingFunction end
@compat abstract type StdPoolingFunction <: PoolingFunction end # built-in poolings

module Pooling
using ..Mocha: StdPoolingFunction

struct Max <: StdPoolingFunction end
struct Mean <: StdPoolingFunction end

end # module Pooling
