using HDF5, JLD

export Snapshot

struct Snapshot <: Coffee
  dir :: AbstractString
end

function init(coffee::Snapshot, ::Net)
  if isdir(coffee.dir)
    m_info("Snapshot directory $(coffee.dir) already exists")
  else
    m_info("Snapshot directory $(coffee.dir) does not exist, creating...")
    mkdir_p(coffee.dir)
  end
end

const SOLVER_STATE_KEY = "solver_state"

function enjoy(lounge::CoffeeLounge, coffee::Snapshot, net::Net, state::SolverState{T}) where {T<:InternalSolverState}
  fn = @sprintf("snapshot-%06d.jld", state.iter)
  m_info("Saving snapshot to $fn...")
  path = joinpath(coffee.dir, fn)
  if isfile(path)
    m_warn("Overwriting $path...")
  end

  jldopen(path, "w") do file
    save_network(file, net)
    write(file, SOLVER_STATE_KEY, snapshot(state))
  end
end
