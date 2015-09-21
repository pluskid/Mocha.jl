using HDF5, JLD
using ArgParse
using Compat

function read_stats(fname)
  stats = jldopen(fname, "r") do file
    read(file, "statistics")
  end
  return stats
end

function number_stats(fnames, names)
  res = Any[]
  for (i, fname) in enumerate(fnames)
    for name in names[i]
      push!(res, (i, fname, name))
    end
  end
  return res
end

function list_stats(numbered_names)
  println("Listing available statistics")
  for (k, (_, fname, name)) in enumerate(numbered_names)
    println("  $k : $fname/$name")
  end
  println("Select statistics to plot using -i and specify the numbers 1-$(length(numbered_names)) seperated with ,")
end

function create_safe_files(fnames, to_tmp)
  # copy to temporary file if requested
  if to_tmp
    stats_files = [tempname() for fname in fnames]
    for (tmpfile,fname) in zip(stats_files, fnames)
      cp(fname, tmpfile)
    end
    return stats_files
  else
    return fnames
  end
end

s = ArgParseSettings()
@add_arg_table s begin
  "--idx", "-i"
    help = "a list of indices seperated by , denoting the statistics that should be plotted"
    arg_type = AbstractString
    default = ""
  "--list", "-l"
    help = "list all available statistics for plotting"
    action = :store_true
  "--tmp", "-t"
    help = "copy the statistics file to a temporary location before plotting (useful when plotting during training)"
    action = :store_true
  "statistics_filenames"
    nargs = '*'
    help = "the filenames of the statistics JLD files"
    required = true
end

# first parse arguments and read statistics files
parsed_args = parse_args(ARGS, s)
filenames = unique(parsed_args["statistics_filenames"])
stats_files = create_safe_files(filenames, parsed_args["tmp"])
all_stats = map(read_stats, stats_files)
# get all unique statistic names that were logged in each files
names = map(keys, all_stats)
# and assign a number to each
numbered_names = number_stats(filenames, names)

# process according to arguments
if parsed_args["list"] || parsed_args["idx"] == ""
  list_stats(numbered_names)
end

using PyPlot
if parsed_args["idx"] != ""
  selected_ind = map(int, split(parsed_args["idx"], ","))
  if any([x < 0 || x > length(numbered_names) for x in selected_ind])
    list_stats(numbered_names)
    error("Invalid index in your list : $selected_ind make sure the indices are between 1 and $(length(numbered_names))")
  end

  figure()
  for ind in selected_ind
    # get the right stats file
    (stats_num, fname, selected) = numbered_names[ind]
    stats = all_stats[stats_num][selected]

    # do the actual plotting
    # x will simply be the iteration number
    #   which we will sort
    x = sort(collect(keys(stats)))
    # and y is the statistics corresponding to
    # the selected statistics you want to plot
    y = [stats[i] for i in x]
    plot(x, y, label="$(fname)/$(selected)")
  end
  legend()

  print("Hit <enter> to continue")
  readline()
  close()
end

# delete temporary file if it was created
if parsed_args["tmp"]
  for f in stats_files
    rm(f)
  end
end
