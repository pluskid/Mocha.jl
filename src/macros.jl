#############################################################
# A convenient macro to define a Composite Type, with
# default values for fields and a constructor that accept
# keyword parameters to initialize the fields. For example
#
# @defstruct MyStruct Any (
#   field1 :: Int = 0,
#   (field2 :: String = "", !isempty(field2))
# )
#
# where each field could be either
#
#   field_name :: field_type = default_value
#
# or put within a tuple, with the second element
# specifying a validation check on the field value.
# In the example above, the default value for
# field2 does not satisfy the assertion, this
# could be used to force user to provide a
# valid value when no meaningful default value
# is available.
#############################################################
macro defstruct(name, super_name, fields)
  if fields.head == :tuple
    fields = fields.args
  else
    fields = [fields]
  end

  field_defs     = Array(Expr, length(fields))     # :(field2 :: Int)
  field_names    = Array(Symbol, length(fields))   # :field2
  field_defaults = Array(Expr, length(fields))     # :(field2 :: Int = 0)
  field_asserts  = Array(Expr, length(fields))     # :(field2 >= 0)

  for i = 1:length(fields)
    field = fields[i]
    if field.head == :tuple
      field_asserts[i] = field.args[2]
      field = field.args[1]
    end
    field_defs[i] = field.args[1]
    field_names[i] = field.args[1].args[1]
    field_defaults[i] = Expr(:kw, field.args...)
  end

  if length(fields) == 0
    # no need to define constructor, empty block
    ctor = Expr(:block)
  else
    # body of layer type, defining fields
    type_body = Expr(:block, field_defs...)

    # constructor
    asserts = map(filter(i -> isdefined(field_asserts,i), 1:length(fields))) do i
      :(@assert($(field_asserts[i])))
    end
    construct = Expr(:call, esc(name), field_names...)
    ctor_body = Expr(:block, asserts..., construct)
    ctor_def = Expr(:call, esc(name), Expr(:parameters, field_defaults...))
    ctor = Expr(:(=), ctor_def, ctor_body)
  end

  quote
    immutable $(esc(name)) <: $super_name
      $type_body
    end

    $ctor
  end
end

#############################################################
# A macro used to characterize a layer. Example
#
# @characterize_layer(HDF5DataLayer,
#     is_source => true,
#     is_sink => false,
#     back_propagate => false,
# )
#############################################################
macro characterize_layer(layer, properties...)
  defs = Array(Expr, length(properties))
  for (i,prop) in enumerate(properties)
    if !(isa(prop, Expr) && prop.head == :(=>))
      error("Property should be: property_name => value")
    end

    prop_name = prop.args[1]
    prop_val  = prop.args[2]
    defs[i] = quote
      $(esc(prop_name))(::$layer) = $prop_val
    end
  end

  Expr(:block, defs...)
end
