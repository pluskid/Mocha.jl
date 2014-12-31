using DataFrames
using Gadfly
using HDF5, JLD

finetune = load("finetune/statistics.jld")["statistics"]
randinit = load("randinit/statistics.jld")["statistics"]
stats = [finetune, randinit]
category = ["pre-train", "random init"]

plot_vars = (("test-accuracy-accuracy", "Accuracy (Test set)"),
             ("obj-val", "Objective Function (Training set)"))

for (var, title) in plot_vars
  dfs = [begin
    data = stats[i][var]
    x = collect(keys(data)); y = [data[xx] for xx in x]
    DataFrame(x=x, y=y, category=category[i])
  end for i = 1:length(stats)]

  df = vcat(dfs...)
  the_plot = plot(df, x="x", y="y", color="category",
      Geom.line, Guide.xlabel("Iteration"), Guide.title(title))

  draw(SVG("$var.svg", 18cm, 9cm), the_plot)
end
