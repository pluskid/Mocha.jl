export ValidationPerformance

type ValidationPerformance <: Coffee
  validation_net :: Net
end

function init(coffee::ValidationPerformance, ::Net)
  init(coffee.validation_net)
end
function enjoy(coffee::ValidationPerformance, ::CoffeeBreakTime.Evening, ::Net, ::SolverState)
  epoch = get_epoch(coffee.validation_net)
  while true
    forward(coffee.validation_net)
    if get_epoch(coffee.validation_net) > epoch
      break
    end
  end

  show_statistics(coffee.validation_net, title="Performance on Validation Set")
  reset_statistics(coffee.validation_net)
end
function destroy(coffee::ValidationPerformance, ::Net)
  # We don't destroy here as we didn't construct the network
  # Let the user destroy the network when they are done.
  # destroy(coffee.validation_net)
end
