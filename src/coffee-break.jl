export CoffeeBreak, check_coffee_break
export CoffeeBreakTime, CoffeeBreakTimeType
export init, enjoy, destroy
export PerformanceOnValidationSet

abstract CoffeeBreakTimeType
module CoffeeBreakTime
using ..Mocha.CoffeeBreakTimeType
type Morning <: CoffeeBreakTimeType end
type Evening <: CoffeeBreakTimeType end
end # module CoffeeBreakTime

# statistics returned by every coffee break
typealias StatsDict Dict{String, Real}

abstract Coffee
function init(::Coffee, ::Net) end
function enjoy(::Coffee, ::CoffeeBreakTimeType, ::Net, ::SolverState) return StatsDict() end
function destroy(::Coffee, ::Net) end

type CoffeeBreak
  coffee        :: Coffee
  every_n_iter  :: Int
  every_n_epoch :: Int
end
function check_coffee_break(cb::CoffeeBreak, t::CoffeeBreakTimeType, state::SolverState, net::Net)
  if cb.every_n_iter > 0
    if state.iter % cb.every_n_iter == 0
      enjoy(cb.coffee, t, net, state)
    end
  elseif cb.every_n_epoch > 0
    if get_epoch(net) % cb.every_n_epoch == 0
      enjoy(cb.coffee, t, net, state)
    end
  end
end

include("coffee/training-summary.jl")
include("coffee/validation-performance.jl")
include("coffee/accumulator.jl")
include("coffee/snapshot.jl")
