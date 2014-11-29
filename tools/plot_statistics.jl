using HDF5, JLD
using ArgParse

function read_stats(fname)
  stats = jldopen(fname, "r") do file
    read(file, "statistics")
  end
  return stats
end

function number_stats(fnames, names)
  res = Dict()
  n = 1
  for (i, fname) in enumerate(fnames)
    for (j, name) in enumerate(names[i])
      res[n] = (i, fname, name)
      n += 1
    end
  end
  return res
end

function list_stats(numbered_names)
  println("Listing available statistics")
  for k in sort(collect(keys(numbered_names)))
    (_, fname, name) = numbered_names[k]
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

get_unique_names(stats) = unique(vcat(map(collect, map(keys, values(stats)))...))

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
  "statistics_filenames"
    nargs = '*'
    help = "the filenames of the statistics hdf5 files"
    required = true
end

# first parse arguments and read statistics files
parsed_args = parse_args(ARGS, s)
filenames = unique(parsed_args["statistics_filenames"])
stats_files = create_safe_files(filenames, parsed_args["tmp"])
all_stats = map(read_stats, stats_files)
# get all unique statistic names that were logged in each files
names = map(get_unique_names, all_stats)
# and assign a number to each 
numbered_names = number_stats(filenames, names)

# process according to arguments
using PyPlot
if parsed_args["list"] || parsed_args["idx"] == ""
  list_stats(numbered_names)
end

if parsed_args["idx"] != ""
  selected_ind = map(int, split(parsed_args["idx"], ","))
  if any([x < 0 || x > length(numbered_names) for x in selected_ind])
    list_stats(numbered_names)
    error("Invalid index in your list : $selected_ind make sure the indices are between 1 and $(length(numbered_names))")
  end

  figure()
  for ind in selected_ind
    (stats_num, fname, key) = numbered_names[ind]
    stats = all_stats[stats_num] 

    N = length(stats)
    x = zeros(N)
    y = zeros(N)
    for (i, iter) in enumerate(sort(collect(keys(stats))))
      x[i] = iter
      y[i] = stats[iter][key]
    end
    plot(x, y, label="$(fname)/$(key)")
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
