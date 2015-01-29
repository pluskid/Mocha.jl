using Coverage

function process_folder(folder="src", ignore=[])
  source_files=Any[]
  filelist = readdir(folder)
  for file in filelist
    if in(file, ignore)
      continue
    end

    fullfile = joinpath(folder,file)
    println(fullfile)
    if isfile(fullfile)
      try
        new_sf = Coveralls.process_file(fullfile)
        push!(source_files, new_sf)
      catch e
        println("Skipped $fullfile")
      end
    else isdir(fullfile)
      append!(source_files, process_folder(fullfile))
    end
  end
  return source_files
end
