using HDF5, JLD

export AccumulateStatistics

type AccumulateStatistics <: Coffee
  modules :: Array{Coffee, 1}
  stats :: Dict{Integer,StatsDict}
  try_load :: Bool
  save :: Bool
  fname :: String

  AccumulateStatistics(modules :: Array{Coffee, 1}; stats = Dict(StatsDict()), try_load = false, save = false, fname = "" ) = new(modules, stats, try_load, save, fname)
end

function init(coffee::AccumulateStatistics, net::Net)
  for m in coffee.modules
    init(m, net)
  end
  if coffee.try_load && isfile(coffee.fname)
    @warn("Statistics file already exists, trying to merge!")
    stats = jldopen(coffee.fname, "r") do file
      read(file, "statistics") 
    end
    merge!(coffee.stats, stats)
  end
end

function enjoy(coffee::AccumulateStatistics, time::CoffeeBreakTime.Morning, net::Net, state::SolverState)
  step = state.iter
  if ! haskey(coffee.stats, step)
    coffee.stats[step] = StatsDict()
  end
  for m in coffee.modules
    merge!(coffee.stats[step], enjoy(m, time, net, state))
  end
end

function enjoy(coffee::AccumulateStatistics, time::CoffeeBreakTime.Evening, net::Net, state::SolverState)
  step = state.iter
  if ! haskey(coffee.stats, step)
    coffee.stats[step] = StatsDict()
  end
  for m in coffee.modules
    merge!(coffee.stats[step], enjoy(m, time, net, state))
  end
  if coffee.save      
    jldopen(coffee.fname, "w") do file
      write(file, "statistics", coffee.stats) 
    end
  end
end
