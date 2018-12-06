export TopologyError

struct TopologyError <: Exception
  desc :: AbstractString
end
Base.showerror(io::IO, e::TopologyError) = print(io, "Illegal Network Topology: ", e.desc)
