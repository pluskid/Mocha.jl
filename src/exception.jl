type TopologyError <: Exception
  desc :: String
end
Base.showerror(io::IO, e::TopologyError) = print(io, "Illegal Network Topology: ", e.desc)
