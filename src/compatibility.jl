using Compat


if VERSION < v"0.3.3"
  function blasfunc(name::Symbol)
    string(name)
  end
elseif VERSION < v"0.5.0-dev+1915"
  function blasfunc(name::Symbol)
    Base.blasfunc(name)
  end
elseif VERSION < v"0.7.0-beta2.199"
  function blasfunc(name::Symbol)
    str_name = string(name)
    fnc_symb = eval(:(Base.BLAS.@blasfunc $str_name))
    fnc_name = string(fnc_symb)
    return fnc_name
  end
else
  function blasfunc(name::Symbol)
    str_name = string(name)
    fnc_symb = eval(:(((Compat.LinearAlgebra).BLAS).@blasfunc($str_name)))
    fnc_name = string(fnc_symb)
    return fnc_name
  end
end
export blasfunc