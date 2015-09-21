function test_adam_solver(backend)
    println("-- Testing simple Adam solver call")
    registry_reset(backend)
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
    data_layer = MemoryDataLayer(batch_size=N, data=Array[X,Y])

    w1 = InnerProductLayer(neuron=Neurons.Sigmoid(), name="ip1", output_dim=20, tops=[:a], bottoms=[:data])
    w2 = InnerProductLayer(neuron=Neurons.Identity(), name="ip2",output_dim=4, tops=[:b], bottoms=[:a])
    loss_layer = SquareLossLayer(name="loss", bottoms=[:b, :label] )
    net = Net("TEST", backend, [w1, w2, loss_layer, data_layer])

    # Make a Solver with max iterations 2
    adam = Adam()
    params = make_solver_parameters(adam, stepsize=0.001,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-8,
                                    max_iter=2)
    solver = Solver(adam, params)
    solve(solver, net)
    destroy(net)
end

if test_cpu
    test_adam_solver(backend_cpu)
end

if test_gpu
    test_adam_solver(backend_gpu)
end
