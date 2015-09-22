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
