export ValidationPerformance

type ValidationPerformance <: Coffee
  validation_net :: Net
end

function init(coffee::ValidationPerformance, ::Net) 
  init(coffee.validation_net)
end
function enjoy(coffee::ValidationPerformance, ::Net, ::SolverState) 
end
function destroy(coffee::ValidationPerformance, ::Net) 
  destroy(coffee.validation_net)
end
