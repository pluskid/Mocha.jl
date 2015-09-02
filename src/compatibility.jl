using Compat

if VERSION < v"0.3.3"
  function blasfunc(name)
    string(name)
  end
else
  function blasfunc(name)
    Base.blasfunc(name)
  end
end

if VERSION < v"0.4-"
  Libdl = Base
else
  Libdl = Base.Libdl
end
