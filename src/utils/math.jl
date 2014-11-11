# julia's built-in int is rounding:
#   int(3.3) => 3
#   int(3.5) => 4
# here we want truncating
#   floorint(3.3) => 3
#   floorint(3.5) => 3
function floorint(x)
  convert(Int, floor(x))
end
