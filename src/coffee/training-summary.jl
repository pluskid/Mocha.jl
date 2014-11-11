export TrainingSummary

type TrainingSummary <: Coffee
end

function enjoy(::TrainingSummary, ::CoffeeBreakTime.Evening, ::Net, state::SolverState)
  summary = @sprintf("%06d :: TRAIN obj-val = %.8f", state.iter, state.obj_val)
  @info(summary)
end
