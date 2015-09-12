using Mocha

using VAE
use_gpu = true
if use_gpu
    backend = GPUBackend()
else
    backend = CPUBackend()
end

# Number of latent variables, size of encoding, decoding layers
# The Kingma & Welling 2013 paper uses 50,500,500 when optimizing likelihood.
# Here we use low dimensional latent space for exploration
net = VAE.make_vae(backend, N_Z, 200, 200)

init(backend)

base_dir = "snapshots_mnist_VAE"

############# Train the model ############

method = Adam()
params = make_solver_parameters(method, max_iter=50000, regu_coef=0.0,
                                lr_policy=LRPolicy.Fixed(0.002),
                                load_from=base_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, TrainingSummary(:iter, :obj_val, :learning_rate, "kl-loss", "bce-loss"), every_n_iter=100)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

solve(solver, net)

############# Play with it ##############

# We use three sliders to choose a value of Z, the latent variable.
# Then we use the network weights to decode this to an image.

using GtkInteract, Winston
# (We could instead use Interact.jl within IJulia Notebook, but it's
#  not yet working with 4.0 Jupyter.)

xx = -1:0.01:1
@manipulate for x in xx, y in xx, z in xx
  imagesc(VAE.latent_to_output([x,y,z]))
end

destroy(net)
shutdown(backend)
