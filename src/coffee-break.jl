export CoffeeBreak, Coffee, CoffeeLounge
export init, enjoy, destroy
export add_coffee_break, check_coffee_break, setup
export update_statistics, get_statistics, save_statistics, shutdown

@compat abstract type Coffee end
function init(::Coffee, ::Net) end
# The first parameter will be a CoffeeLounge, we put Any here because
# Julia do not have forward declaration
function enjoy(::Any, ::Coffee, ::Net, ::SolverState) end
function destroy(::Coffee, ::Net) end

struct CoffeeBreak
  coffee        :: Coffee
  every_n_iter  :: Int
  every_n_epoch :: Int
end


############################################################
# Coffee break utilities
############################################################
#-- This function is to be called by the end-user
function setup_coffee_lounge(solver::Solver; save_into::AbstractString="", every_n_iter::Int=1, file_exists=:merge)
  solver.coffee_lounge.filename=save_into
  solver.coffee_lounge.save_every_n_iter=every_n_iter
  solver.coffee_lounge.file_exists=file_exists
end

function add_coffee_break(solver::Solver, coffee::Coffee; kw...)
  add_coffee_break(solver.coffee_lounge, coffee; kw...)
end

################################################################################
# Coffee Lounge
################################################################################
using HDF5, JLD

const StatisticsValue = AbstractFloat
const StatisticsRecords = Dict{Int, StatisticsValue}
mutable struct CoffeeLounge
  filename          :: AbstractString
  save_every_n_iter :: Int
  file_exists       :: Symbol # :overwrite, :panic, :merge

  statistics        :: Dict{AbstractString, StatisticsRecords}
  coffee_breaks     :: Vector{CoffeeBreak}

  stats_modified    :: Bool
  last_epoch        :: Int
  curr_iter         :: Int

  CoffeeLounge(;filename="", save_every_n_iter=1, file_exists=:merge) = begin
    lounge = new(filename, save_every_n_iter, file_exists)
    lounge.statistics = Dict{AbstractString, StatisticsRecords}()
    lounge.coffee_breaks = CoffeeBreak[]
    return lounge
  end
end

function setup(lounge::CoffeeLounge, state::SolverState, net::Net)
  if !isempty(lounge.filename)
    directory = dirname(lounge.filename)
    if !isempty(directory) && !isdir(directory)
      mkdir_p(directory)
    end

    if isfile(lounge.filename)
      if lounge.file_exists == :overwrite
        m_warn("Overwriting existing coffee lounge statistics in $(lounge.filename)")
      elseif lounge.file_exists == :merge
        m_info("Merging existing coffee lounge statistics in $(lounge.filename)")
        lounge.statistics = jldopen(lounge.filename) do file
          read(file, "statistics")
        end
      elseif lounge.file_exists == :panic
        error("File already exists for coffee lounge statistics store: $(lounge.filename)")
      else
        error("Coffee lounge: unknown behavior on file exists: $(lounge.file_exists)")
      end
    end
  end

  lounge.stats_modified = false
  lounge.last_epoch = get_epoch(net)
  lounge.curr_iter = state.iter

  for cb in lounge.coffee_breaks
    init(cb.coffee, net)
  end
end

function update_statistics(dummy, key::AbstractString, val::StatisticsValue)
  # dummy function used when you do not want to record statistics
end

function update_statistics(lounge::CoffeeLounge, key::AbstractString, val::StatisticsValue)
  dict = get(lounge.statistics, key, StatisticsRecords())
  dict[lounge.curr_iter] = val
  lounge.statistics[key] = dict
  lounge.stats_modified = true
end

function get_statistics(lounge::CoffeeLounge, key::AbstractString)
  if haskey(lounge.statistics, key)
    lounge.statistics[key]
  else
    StatisticsRecords()
  end
end

function save_statistics(lounge::CoffeeLounge)
  if !isempty(lounge.filename) && lounge.stats_modified
    jldopen(lounge.filename, "w") do file
      write(file, "statistics", lounge.statistics)
    end
    lounge.stats_modified = false
  end
end

function shutdown(lounge::CoffeeLounge, net::Net)
  for cb in lounge.coffee_breaks
    destroy(cb.coffee, net)
  end

  save_statistics(lounge)
end

function add_coffee_break(lounge::CoffeeLounge, coffee::Coffee; every_n_iter::Int=0, every_n_epoch::Int=0)
  cb = CoffeeBreak(coffee, every_n_iter, every_n_epoch)
  push!(lounge.coffee_breaks, cb)
end

function check_coffee_break(lounge::CoffeeLounge, state::SolverState, net::Net)
  lounge.curr_iter = state.iter

  for cb in lounge.coffee_breaks
    if cb.every_n_iter > 0
      if state.iter % cb.every_n_iter == 0
        enjoy(lounge, cb.coffee, net, state)
      end
    elseif cb.every_n_epoch > 0
      epoch = get_epoch(net)
      if epoch != lounge.last_epoch && epoch % cb.every_n_epoch == 0
        enjoy(lounge, cb.coffee, net, state)
      end
    end
  end

  if state.iter % lounge.save_every_n_iter == 0
    save_statistics(lounge)
  end

  lounge.last_epoch = get_epoch(net)
end

include("coffee/training-summary.jl")
include("coffee/validation-performance.jl")
include("coffee/snapshot.jl")
