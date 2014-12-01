export TrainingSummary

type TrainingSummary <: Coffee
end

function enjoy(lounge::CoffeeLounge, ::TrainingSummary, ::CoffeeBreakTime.Evening, ::Net, state::SolverState)
  summary = @sprintf("%06d :: TRAIN obj-val = %.8f", state.iter, state.obj_val)
  @info(summary)

  update_statistics(lounge, "obj-val", state.obj_val)
end
