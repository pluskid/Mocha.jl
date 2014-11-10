export CoffeeBreak, go
export PerformanceOnValidationSet

abstract Coffee
function enjoy(::CoffeeBreak)
  error("Not implemented (should enjoy coffee)")
end

type CoffeeBreak
  coffee        :: Coffee
  every_n_iter  :: Int
  every_n_epoch :: Int

  CoffeeBreak(coffee::Coffee; every_n_iter=0, every_n_epoch=0) =
      CoffeeBreak(coffee, every_n_iter, every_n_epoch)
end
function check_coffee_break(state::SolverState, net::Net, cb::CoffeeBreak)
  if cb.every_n_iter > 0
    if state.iter % cb.every_n_iter == 0
      enjoy(cb.coffee)
    end
  elseif cb.every_n_epoch
    if get_epoch(net) % cb.every_n_epoch == 0
      enjoy(cb.coffee)
    end
  end
end


type PerformanceOnValidationSet <: Coffee
  net :: Net
end

