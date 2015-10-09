using Mocha

include("VAE.jl")

backend = DefaultBackend()
init(backend)

# Number of latent variables, size of encoding, decoding layers
# The Kingma & Welling 2013 paper uses 50,500,500 when optimizing likelihood.
# Here we use low dimensional latent space for exploration
N_Z = 50
net = make_vae(backend, N_Z, 400, 400)


############# Train the model ############

base_dir = "snapshots_mnist_VAE_exp_50_400"

method = Adam()
params = make_solver_parameters(method, max_iter=20000, regu_coef=0.0,
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.5),
                                #lr_policy=LRPolicy.Fixed(0.002),
                                load_from=base_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, TrainingSummary(:iter, :obj_val, :learning_rate, "kl-loss", "bce-loss"), every_n_iter=100)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

solve(solver, net)



############# Play with it ##############

# We use three sliders to choose a value of Z, the latent variable.
# Then we use the network weights to decode this to an image.

using Interact, Winston
# (Need a recent Interact.jl on Ipython 4/Jupyter - https://github.com/JuliaLang/Interact.jl/issues/73)
# Pkg.checkout("Interact.jl"), and make sure ipywidgets Python package is installed.

latent_to_output = make_latent_to_output(net, 12, 13)

xx = -1:0.01:1
@Interact.manipulate for x in xx, y in xx, z in xx
  imagesc(latent_to_output([x,y,z]))
end

destroy(net)
shutdown(backend)
