export TrainingSummary

type TrainingSummary <: Coffee
end

function enjoy(lounge::CoffeeLounge, ::TrainingSummary, ::Net, state::SolverState)
  # we do not report objective value at iteration 0 because it has not been computed yet
  summary = @sprintf("%06d :: TRAIN obj-val = %.8f :: LR = %8f :: Mom = %8f", state.iter, state.obj_val, state.learning_rate, state.momentum)
  @info(summary)

  update_statistics(lounge, "obj-val", state.obj_val)
end
