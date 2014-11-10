export CoffeeBreak, check_coffee_break
export init, enjoy, destroy
export PerformanceOnValidationSet

abstract Coffee
function init(::Coffee, ::Net) end
function enjoy(::Coffee, ::Net, ::SolverState) end
function destroy(::Coffee, ::Net) end

type CoffeeBreak
  coffee        :: Coffee
  every_n_iter  :: Int
  every_n_epoch :: Int
end
function check_coffee_break(cb::CoffeeBreak, state::SolverState, net::Net)
  if cb.every_n_iter > 0
    if state.iter % cb.every_n_iter == 0
      enjoy(cb.coffee, net, state)
    end
  elseif cb.every_n_epoch > 0
    if get_epoch(net) % cb.every_n_epoch == 0
      enjoy(cb.coffee, net, state)
    end
  end
end

include("coffee/training-summary.jl")
include("coffee/validation-performance.jl")
