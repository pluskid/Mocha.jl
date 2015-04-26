export TrainingSummary

type TrainingSummary <: Coffee
  show_iter :: Bool
  show_obj_val :: Bool
  show_lr :: Bool
  show_mom:: Bool
  
  #Default Constructor
  function TrainingSummary(;show_iter=true,show_obj_val=true,show_lr=false,show_mom=false)
    newTrainingSummary = new(show_iter,show_obj_val,show_lr,show_mom)

    return newTrainingSummary
  end

end

function enjoy(lounge::CoffeeLounge, coffee::TrainingSummary, ::Net, state::SolverState)
  # we do not report objective value at iteration 0 because it has not been computed yet
  if coffee.show_iter
    txt_iter = @sprintf("ITER = %06d",state.iter)
  else
    txt_iter = ""
  end

  if coffee.show_obj_val
    txt_obj_val = @sprintf(":: TRAIN obj-val = %.8f",state.obj_val)
    update_statistics(lounge, "obj-val", state.obj_val)
  else
    txt_obj_val = ""
  end

  if coffee.show_lr
    txt_lr = @sprintf(":: LR = %.8f",state.learning_rate)
    update_statistics(lounge, "learning-rate", state.learning_rate)
  else
    txt_lr = ""
  end

  if coffee.show_mom
    txt_mom = @sprintf(":: MOM = %.8f",state.momentum)
    update_statistics(lounge, "momentum", state.momentum)
  else
    txt_mom = ""
  end

  summary = string(txt_iter,txt_obj_val,txt_lr,txt_mom)
  
  @info(summary)
end
