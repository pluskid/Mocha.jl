#############################################################
# A convenient macro to define a Composite Type, with
# default values for fields and a constructor that accept
# keyword parameters to initialize the fields. For example
#
# @defstruct MyStruct Any (
#   field1 :: Int = 0,
#   (field2 :: AbstractString = "", !isempty(field2))
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
#
# The macro will define a constructor that could accept
# the keyword arguments. Since the defined type is
# immutable, a "copy" function is also defined, which
# takes a prototype object, and accept extra keyword
# parameters that could be used to construct a new
# object with specified changes of fields.
#############################################################
import Base.copy
export      copy

macro defstruct(name, super_name, fields)
  @assert fields.head == :tuple
  fields = fields.args
  @assert length(fields) > 0
  name = esc(name)

  field_defs     = Array{Expr}(length(fields))     # :(field2 :: Int)
  field_names    = Array{Symbol}(length(fields))   # :field2
  field_defaults = Array{Expr}(length(fields))     # :(field2 :: Int = 0)
  field_asserts  = Array{Expr}(length(fields))     # :(field2 >= 0)

  for i = 1:length(fields)
    field = fields[i]
    if field.head == :tuple
      field_asserts[i] = field.args[2]
      field = field.args[1]
    end
    field_defs[i] = esc(field.args[1])
    field_names[i] = field.args[1].args[1]
    field_defaults[i] = Expr(:kw, field.args...)
  end

  # body of layer type, defining fields
  type_body = Expr(:block, field_defs...)

  # constructor
  asserts = map(filter(i -> isassigned(field_asserts,i), 1:length(fields))) do i
    :(@assert($(field_asserts[i])))
  end
  construct = Expr(:call, name, field_names...)
  ctor_body = Expr(:block, asserts..., construct)
  ctor_def = Expr(:call, name, Expr(:parameters, field_defaults...))
  ctor = Expr(:(=), ctor_def, ctor_body)

  # for copy constructor
  field_assigns = Expr(:block, [:(params[symbol($(esc(string(fname))))] = proto.$fname) for fname in field_names]...)
  field_expose = Expr(:block, [:($(esc(fname)) = params[symbol($(esc(string(fname))))]) for fname in field_names]...)
  assert_block = Expr(:block, asserts...)
  obj_construct = Expr(:call, name, field_names...)
  copy_fname = esc(:copy)

  quote
    immutable $(name) <: $super_name
      $type_body
    end

    $ctor

    function $copy_fname(proto::$name; kw...)
      params = Dict()
      $field_assigns

      for (k,v) in kw
        @assert haskey(params, k) "Unrecognized field " * string(k) * " for " * $(string(name.args[1]))
        params[k] = v
      end

      $field_expose
      $assert_block

      $obj_construct
    end
  end
end

@static if VERSION < v"0.6-"
  function parse_property(prop)
    @assert(isa(prop, Expr) && prop.head == :(=>), "Property should be: property_name => value")
    prop.args[1], prop.args[2]
  end
else
  function parse_property(prop)
    @assert(isa(prop, Expr) && prop.head == :(call) && prop.args[1] == :(=>), "Property should be: property_name => value")
    prop.args[2], prop.args[3]
  end
end

#############################################################
# A macro used to characterize a layer. Example
#
# @characterize_layer(HDF5DataLayer,
#     is_source => true,
#     is_sink => false,
#     can_do_bp => false,
# )
#############################################################
macro characterize_layer(layer, properties...)
  defs = Array{Expr}(length(properties))
  for (i,prop) in enumerate(properties)
    prop_name, prop_val = parse_property(prop)
    defs[i] = quote
      $(esc(prop_name))(::$(esc(layer))) = $prop_val
    end
  end

  Expr(:block, defs...)
end
