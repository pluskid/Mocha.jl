export Layer, LayerState, AbstractParameter
export copy_solver_state!

@compat abstract type Layer end
@compat abstract type LayerState end

# forward declaration
@compat abstract type AbstractParameter end
