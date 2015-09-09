export TrainingSummary

type TrainingSummary <: Coffee
  statistic_names :: Vector{Symbol}

  #Default Constructor
  TrainingSummary(statistic_names...) = begin
      if !isempty(statistic_names)
          new(collect(statistic_names))
      else
          new([:iter, :obj_val])
      end
  end
end


function enjoy(lounge::CoffeeLounge, coffee::TrainingSummary, ::Net, state::SolverState)
  summaries = String[]

  for statistic_name in coffee.statistic_names
    statval = get_statistic(state, statistic_name)
    push!(summaries, format_statistic(state, statistic_name, statval))
    if statistic_name != :iter
        update_statistics(lounge, string(statistic_name), statval)
    end
  end

  @info(" TRAIN ", join(summaries, ' '))

end
