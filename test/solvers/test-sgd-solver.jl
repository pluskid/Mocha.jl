function test_sgd_solver(backend)
    println("-- Testing simple SGD solver call")
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

    w1 = InnerProductLayer(neuron=Neurons.Sigmoid(), name="ip1",output_dim=20, tops=[:a], bottoms=[:data])
    w2 = InnerProductLayer(neuron=Neurons.Identity(), name="ip2",output_dim=4, tops=[:b], bottoms=[:a])
    loss_layer = SquareLossLayer(name="loss", bottoms=[:b, :label] )


    net = Net("TEST", backend, [w1, w2, loss_layer, data_layer])

    # Make a Solver with max iterations 2
    sgd = SGD()
    params = make_solver_parameters(sgd,
                                    max_iter=2,
                                    mom_policy=MomPolicy.Fixed(0.9),
                                    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                    load_from="")
    solver = Solver(sgd, params)
    solve(solver, net)
    # TODO check gradient updates
    # TODO check snapshot loading
    # TODO check statistic saving
    # TODO check other lr policies
    # TODO check other mom policies
    destroy(net)
end

if test_cpu
    test_sgd_solver(backend_cpu)
end

if test_gpu
    test_sgd_solver(backend_gpu)
end
