export ValidationPerformance

type ValidationPerformance <: Coffee
  validation_net :: Net
end

function init(coffee::ValidationPerformance, ::Net) 
  init(coffee.validation_net)
end
function enjoy(coffee::ValidationPerformance, ::Net, ::SolverState) 
  epoch = get_epoch(coffee.validation_net)
  while true
    forward(coffee.validation_net)
    if get_epoch(coffee.validation_net) > epoch
      break
    end
  end

  show_statistics(coffee.validation_net)
  reset_statistics(coffee.validation_net)
end
function destroy(coffee::ValidationPerformance, ::Net) 
  destroy(coffee.validation_net)
end
