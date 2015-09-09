using HDF5, JLD

export Snapshot

type Snapshot <: Coffee
  dir :: String
end

function init(coffee::Snapshot, ::Net)
  if isdir(coffee.dir)
    @info("Snapshot directory $(coffee.dir) already exists")
  else
    @info("Snapshot directory $(coffee.dir) does not exist, creating...")
    mkdir_p(coffee.dir)
  end
end

const SOLVER_STATE_KEY = "solver_state"

function enjoy{T<:InternalSolverState}(lounge::CoffeeLounge, coffee::Snapshot, net::Net, state::SolverState{T})
  fn = @sprintf("snapshot-%06d.jld", state.iter)
  @info("Saving snapshot to $fn...")
  path = joinpath(coffee.dir, fn)
  if isfile(path)
    @warn("Overwriting $path...")
  end

  jldopen(path, "w") do file
    save_network(file, net)
    write(file, SOLVER_STATE_KEY, snapshot(state))
  end
end
