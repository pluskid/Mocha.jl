using HDF5, JLD
using ArgParse

function read_stats(fname)
  stats = jldopen(fname, "r") do file
    read(file, "statistics")
  end
  return stats
end

function list_statistics(names)
  println("Listing available statistics")
  for (i, name) in enumerate(names)
      println("  $i : $name : $i")
  end
  println("Select statistics to plot using -i and specify the numbers 1-$(length(names)) seperated with ,")
end

function create_safe_file(fname, to_tmp)
  # copy to temporary file if requested
  if to_tmp
    stats_file = tempname()
    cp(fname, stats_file)
    return stats_file
  else
    return fname
  end
end

s = ArgParseSettings()
@add_arg_table s begin
  "--idx", "-i"
    help = "a list of indices seperated by , denoting the statistics that should be plotted"
    arg_type = String
    default = ""
  "--list", "-l"
    help = "list all available statistics for plotting"
    action = :store_true
  "--tmp", "-t"
    help = "copy the statistics file to a temporary location before plotting (useful when plotting during training)"
    action = :store_true
  "statistics_filename"
    help = "the filename of the statistics hdf5 file"
    required = true
end

# first parse arguments and read statistics file
parsed_args = parse_args(ARGS, s)
stats_file = create_safe_file(parsed_args["statistics_filename"], parsed_args["tmp"])
stats = read_stats(stats_file)
# get all unique statistic names that were logged
names = unique(map(collect, map(keys, values(stats))))[1]

# process according to arguments
using PyPlot
if parsed_args["list"] || parsed_args["idx"] == ""
  list_statistics(names)
end

if parsed_args["idx"] != ""
  selected_ind = map(int, split(parsed_args["idx"], ","))
  if any([x < 0 || x > length(names) for x in selected_ind])
    list_statistics(names)
    error("Invalid index in your list : $selected_ind make sure the indices are between 1 and $(length(names))")
  end

  selected = [names[i] for i in selected_ind]
    
  figure()
  for key in selected
    N = length(stats)
    x = zeros(N)
    y = zeros(N)
    for (i, iter) in enumerate(sort(collect(keys(stats))))
      x[i] = iter
      y[i] = stats[iter][key]
    end
    plot(x, y, label=key)
  end
  legend()
   
  print("Hit <enter> to continue")
  readline()
  close()
end

# delete temporary file if it was created
if parsed_args["tmp"]
  rm(stats_file)
end
