# visualize a net with GraphViz, by outputing dot description file
function net2dot(io::IO, net::Net)
  println(io, """digraph "$(net.name)" {""")
  println(io, """rankdir = BT;""")
  println(io, """node [shape=record,fontname="Fira Mono"];""")

  #------------------------------------------------------------
  # define layers
  for i = 1:length(net.layers)
    layer = net.layers[i]

    layer_type = replace("$(typeof(layer))", r"Layer$", "")
    layer_name = layer.name
    neuron_name = ""
    if in(:neuron, names(layer)) && !isa(layer.neuron, Neurons.Identity)
      neuron_name = "\\n($(typeof(layer.neuron)))"
    end

    # layer name
    print(io, """layer$i [label="$layer_type\\n$layer_name$neuron_name|{top:|bot:}|""")
    print(io, """{""")

    # top blobs
    if in(:tops, names(layer))
      print(io, """{""")
      for j = 1:length(layer.tops)
        if j != 1
          print(io, "|")
        end
        print(io, "<t$j> $(layer.tops[j])")
        print(io, " ($(join(map(x -> "$x", size(net.states[i].blobs[j])), "x")))")
      end
      print(io, """}""")
    end

    print(io, """|""")

    # bottom blobs
    if in(:bottoms, names(layer))
      print(io, """{""")
      for j = 1:length(layer.bottoms)
        if j != 1
          print(io, "|")
        end
        print(io, "<b$j> $(layer.bottoms[j])")
        print(io, " ($(join(map(x -> "$x", size(net.blobs_forward[i][j])), "x")))")
      end
      print(io, """}""")
    end

    println(io, """}"];""")
  end

  #------------------------------------------------------------
  # Connect blobs
  top_maps = Dict{Symbol,String}()
  for i = 1:length(net.layers)
    if in(:tops, names(net.layers[i]))
      for j = 1:length(net.layers[i].tops)
        top_maps[net.layers[i].tops[j]] = "layer$i:t$j"
      end
    end
  end
  for i = 1:length(net.layers)
    layer = net.layers[i]
    if in(:bottoms, names(layer))
      for j = 1:length(layer.bottoms)
        println(io, "$(top_maps[layer.bottoms[j]]) ->layer$i:b$j;")
      end
    end
  end

  println(io, "}") # digraph
end
