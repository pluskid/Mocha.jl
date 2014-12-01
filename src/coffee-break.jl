export CoffeeBreak, check_coffee_break
export CoffeeBreakTime, CoffeeBreakTimeType
export init, enjoy, destroy

abstract CoffeeBreakTimeType
module CoffeeBreakTime
using ..Mocha.CoffeeBreakTimeType
type Morning <: CoffeeBreakTimeType end
type Evening <: CoffeeBreakTimeType end
end # module CoffeeBreakTime


abstract Coffee
function init(::Coffee, ::Net) end
# The first parameter will be a CoffeeLounge, we put Any here because
# Julia do not have forward declaration
function enjoy(::Any, ::Coffee, ::CoffeeBreakTimeType, ::Net, ::SolverState) end
function destroy(::Coffee, ::Net) end

type CoffeeBreak
  coffee        :: Coffee
  every_n_iter  :: Int
  every_n_epoch :: Int
end

################################################################################
# Coffee Lounge
################################################################################
using HDF5, JLD

typealias StatisticsRecords Dict{Int, FloatingPoint}
type CoffeeLounge
  filename :: String
  save_every_n_iter :: Int
  file_exists :: Symbol # :overwrite, :panic, :merge

  statistics :: Dict{Symbol, StatisticsRecords}
  coffee_breaks :: Vector{CoffeeBreak}

  stats_modified :: Bool
  last_epoch :: Int

  file :: JLD.JldFile

  CoffeeLounge(;filename="", save_every_n_iter=1, file_exists=:merge) = begin
    lounge = new(filename, save_every_n_iter, file_exists)
    lounge.statistics = Dict{Symbol, StatisticsRecords}()
    lounge.coffee_breaks = CoffeeBreak[]
    return lounge
  end
end

function setup(lounge::CoffeeLounge, state::SolverState, net::Net)
  if !isempty(lounge.filename)
    if isfile(lounge.filename)
      if lounge.file_exists == :overwrite
        @warn("Overwriting existing coffee lounge statistics in $(lounge.filename)")
        lounge.file = jldopen(lounge.filename, "w")
      elseif lounge.file_exists == :merge
        @info("Merging existing coffee lounge statistics in $(lounge.filename)")
        lounge.statistics = jldopen(lounge.filename) do file
          read(file, "statistics")
        end
        lounge.file = jldopen(lounge.filename, "w")
      elseif lounge.file_exists == :panic
        error("File already exists for coffee lounge statistics store: $(lounge.filename)")
      else
        error("Coffee lounge: unknown behavior on file exists: $(lounge.file_exists)")
      end
    else
      lounge.file = jldopen(lounge.filename, "w")
    end
  end

  lounge.stats_modified = false
  lounge.last_epoch = get_epoch(net)

  for cb in lounge.coffee_breaks
    init(cb.coffee, net)
  end
end

function update_statistics(lounge::CoffeeLounge, key::Symbol, stats::(Int, Any)...)
  dict = get(lounge.statistics, key, StatisticsRecords())
  for (k,v) in stats
    dict[k] = v
  end
  lounge.statistics[key] = dict
  lounge.stats_modified = true
end

function save_statistics(lounge::CoffeeLounge)
  if !isempty(lounge.filename) && lounge.stats_modified
    lounge.file["statistics"] = lounge.statistics
    lounge.stats_modified = false
  end
end

function shutdown(lounge::CoffeeLounge)
  for cb in solver.coffee_breaks
    destroy(cb.coffee, net)
  end

  if !isempty(lounge.filename)
    save_statistics(lounge)
    close(lounge.file)
  end
end

function check_coffee_break(lounge::CoffeeLounge, t::CoffeeBreakTimeType, state::SolverState, net::Net)
  for cb in lounge.coffe_breaks
    if cb.every_n_iter > 0
      if state.iter % cb.every_n_iter == 0
        enjoy(lounge, cb.coffee, t, net, state)
      end
    end

    if cb.every_n_epoch > 0
      epoch = get_epoch(net)
      if epoch != lounge.last_epoch && epoch % cb.every_n_epoch == 0
        enjoy(lounge, cb.coffee, t, net, state)
      end
    end
  end

  if t == CoffeeBreakTime.Evening
    if state.iter % lounge.save_every_n_iter == 0
      save_statistics(lounge)
    end

    lounge.last_epoch = get_epoch(net)
  end
end

include("coffee/training-summary.jl")
include("coffee/validation-performance.jl")
include("coffee/accumulator.jl")
include("coffee/snapshot.jl")
