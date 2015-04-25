export TrainingSummary

type TrainingSummary <: Coffee
  dispIter :: Bool
  dispObj_val :: Bool
  dispLR :: Bool
  dispMom:: Bool
  
  #Default Constructor
  #Put the "string" parameter to make the rest be optional
  function TrainingSummary(string="dontuseme";showIter=true,showObj_val=true,showLR=false,showMom=false)
    newTrainingSummary = new(showIter,showObj_val,showLR,showMom)

    return newTrainingSummary
  end

  TrainingSummary() = new(true,true,false,false)

end

function enjoy(lounge::CoffeeLounge, coffee::TrainingSummary, ::Net, state::SolverState)
  # we do not report objective value at iteration 0 because it has not been computed yet
  if coffee.dispIter
    txtIter = @sprintf("ITER = %06d",state.iter)
  else
    txtIter = ""
  end

  if coffee.dispObj_val
    txtObj_val = @sprintf(":: TRAIN obj-val = %.8f",state.obj_val)
  else
    txtObj_val = ""
  end

  if coffee.dispLR
    txtLR = @sprintf(":: LR = %.8f",state.learning_rate)
  else
    txtLR = ""
  end

  if coffee.dispMom
    txtMom = @sprintf(":: MOM = %.8f",state.momentum)
  else
    txtMom = ""
  end

  summary = string(txtIter,txtObj_val,txtLR,txtMom)
  
  @info(summary)

  update_statistics(lounge, "obj-val", state.obj_val)
end
