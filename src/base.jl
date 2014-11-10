export Layer, LayerState

abstract Layer      # define layer type, parameters
abstract LayerState # hold layer state, filters


export Solver, SolverState

abstract Solver
type SolverState
  iter    :: Int
  obj_val :: Float64
end

