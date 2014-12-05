export Layer, LayerState, AbstractParameter

abstract Layer
abstract LayerState

# forward declaration
abstract AbstractParameter

export Solver, SolverState

abstract Solver
type SolverState
  iter          :: Int
  obj_val       :: Float64
  learning_rate :: Float64
  momentum      :: Float64

  SolverState() = new(0, 0.0, 0.0, 0.0)
end
function copy_solver_state!(dst::SolverState, src::SolverState)
  dst.iter = src.iter
  dst.obj_val = src.obj_val
end
