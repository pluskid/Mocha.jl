#############################################################
# steven varga 2015 march, Toronto
# licence: MIT Licence
#############################################################

###########################################################
# pretty print result layer by layer
#
import Base.show
export test_gradients

function show(model::Net, e )
    #show( model )
    println("\n*************************************************")
    println("gradient check with finite difference method")
    println("passed: '.'   failed: 'x' ")
    println("*************************************************")
    pos = 1
    for l = 1:length(model.layers)
        if Mocha.has_param(model.layers[l])
            println()
            println( model.layers[l] )
            println( "---------------------------------------------------" )
            w = model.states[l].W.data
            b = model.states[l].b.data
            W = reshape( e[pos:(pos+length(w)-1)], size(w) );
            pos += length(w);  println("weights:");show( W ); println("\n bias:")
            B = reshape( e[pos:(pos+length(b)-1)], size(b) );
            pos += length(b); show( B' );  println()
        end
    end
end


################################################################
# model paramter unrolling into a vector
# suboptimal;
# optimal implemantation constructs model params in a cont.
# space, where unrolling becomes trivial
# it left as an exercise for the reader
function unroll_parameters( model::Net )
    theta = Array{Float64}() # initial state is one length ??? weird
    for l = 1:length(model.layers)
        if Mocha.has_param(model.layers[l])

            for m in model.states[l].parameters
                θ = m.blob.data
                size =  length(θ)
                theta = [ theta; reshape(θ, size, 1)]
            end
        end
    end
    # note the begining, work around the first element
    # is fluke
    return theta[2:end]
end

function unroll_gradients( model::Net )
    theta = Array{Float64}() # initial state is one length ??? weird
    for l = 1:length(model.layers)
        if Mocha.has_param(model.layers[l])
            for m in model.states[l].parameters
                θ = m.gradient.data
                size =  length(θ)
                theta = [ theta; reshape(θ, size, 1)]
            end
        end
    end
    # note the begining, work around as the first element
    # is fluke
    return theta[2:end]
end

#################################################
# updates model paramaters by copying
#
function update_θ!(model,θᵢ)
    pos = 1
    for l = 1:length(model.layers)
        if Mocha.has_param(model.layers[l])

            for m in model.states[l].parameters
                θ = m.blob.data
                len =  length(θ)
                Base.copy!(θ, reshape( θᵢ[pos:(pos+len-1)], size(θ) ))
                pos += len
            end
        end
    end

end


# (J,θ,▽) =
function  hypothesis_and_gradient( model::Net )
  θ = unroll_parameters(model)
  # set_model_params!(model,θ)
  # use closure to embedd params
  function J( θᵢ )
    update_θ!(model,θᵢ)           # update model params
    forward( model )              # compute cost
    return model.states[end].loss # which is stored in the last node or model state
  end

  function grad!(θᵢ,∇ᵢ)
    update_θ!(model,θᵢ)           # update model parameters
    backward(model)               # compute gradients
    ∇ = unroll_gradients(model)   # retrieve them from model state
    Base.copy!(∇ᵢ,∇)              # and update them
  end

  return (J,θ,grad!)
end


#########################################################
#
# J cost function
# g! computes and updates the gradient of the model given θ model params
function compute_finite_difference( J::Function, g!::Function, θ::Vector{Float64}; ε = 1e-6, digit=8 )
  ∇ᵋ = similar(θ);  # gradient/slope with 2ε wiggle room
  ∇  = similar(θ);  # gradient/slope at midpoint
  # compute C cost and ∇ gradient at mid point
  C = J(θ);  	 g!(θ,∇);

  ## define the two sided derivative
  θ⁺ = similar(θ); θ⁻ = similar(θ);

  # iterate through cost function and calculate slope
  for i=1:length(θ)
    Base.copy!(θ⁺,θ);  θ⁺[i] = θ⁺[i] + ε;
    Base.copy!(θ⁻,θ);  θ⁻[i] = θ⁻[i] - ε;
    ∇ᵋ[i] = ( J(θ⁺) - J(θ⁻) ) / 2ε
  end
  return (∇ᵋ,∇)
end

# ############################################################
# gradient check with two sided finite difference method
# based on prof. Andrew Ng,  Machine learning, Coursera
#
#  -- this part could use more work, as an example if gradient fails
#  indicate the positon in the layer itself; to aid debugging.  Since author
#  ported this code from his implentation, further fitting is needed

function gradient_check(model::Net, epsilon::Float64, digit::Int, visual::Bool)
    # create objective that computes grad( θ ), and cost( θ )
    (J,θ, grad) = hypothesis_and_gradient( model::Net )
    ∇ᵋ,∇ = compute_finite_difference( J, grad, θ )

    # do actual comparison with `digit` numerical percision
    # ∇⁺ = round(∇⁺, 4); 	∇ = round(∇, 4)
    idx = round.( abs.(∇ᵋ - ∇), digit ) .!= 0
    if visual
        δ = Array{Char}(length(idx));  fill!(δ,'.')
        δ[idx] = 'x'
        show(model, δ)
        #show(model,round(∇ᵋ,digit) ); show(model,round(∇,digit))
    end
    # return false if fail at any point
    # TODO: check if correct
    sum( round.( abs.(∇ᵋ - ∇), digit) ) < epsilon
end


###############################################################
# exported method
#
# make sure to set datalayer so the total size is same as  batch_size
# one line of data+label is sufficient; but works with minibatches as well
#
# your milage may vary; function of epsilon ~ 1e-6 - 1e-10
# signifacant digit 6 - 8 higher is better gradient
# if your fanout is big theninitial weights may be set small
# when gradient fails reduce fan-out and test again; your
# gradients may be ok
###############################################################
function test_gradients(net::Net; epsilon=1e-6, digit=6, visual=true )
    return  typeof(net.backend) == Mocha.CPUBackend  ?
         gradient_check( net, epsilon, digit, visual ) :false
end



#= test driver; used code from existing example
detach and remove leading closing comment

use_cuda = false

using Mocha
srand(12345678)

############################################################
# Prepare Random Data
############################################################
N = 5    # works with arbitrary minibatch size as long as
         # N == batch_size in MemoryDataLayer so it cycles through
         # and gets the same data during forward()
M = 10
P = 4

X = rand(M, N)
W = rand(M, P)
B = rand(P, 1)

Y = (W'*X .+ B)
Y = Y + 0.01*randn(size(Y))

############################################################
# Define network
############################################################
 backend = CPUBackend()
init(backend)

data_layer = MemoryDataLayer(batch_size=N, data=Array[X,Y])

w1 = InnerProductLayer(neuron=Neurons.Sigmoid(), name="ip1",output_dim=20, tops=[:a], bottoms=[:data])
w2 = InnerProductLayer(neuron=Neurons.Identity(), name="ip2",output_dim=4, tops=[:b], bottoms=[:a])
loss_layer = SquareLossLayer(name="loss", bottoms=[:b, :label] )


net = Net("TEST", backend, [w1,w2, loss_layer, data_layer])
println(net)



# epsilon:     milage may vary 1e-4 - 1e-8
# digit:       compare this many digits to check for 'identity'
# visualize:   prints out correct and failed positions of gradients
test_gradients(net, epsilon=1e-8, digit=6, visual=true )

shutdown(backend)
=#
