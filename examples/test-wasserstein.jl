use_cuda = true
data_type = Float64

if use_cuda
  ENV["MOCHA_USE_CUDA"] = "true"
end

using Mocha
srand(123)

N = 1000
d = 4
lambda = 0.1
sinkhorn_iter = 10

n_hidden_layer = 2
n_hidden_unit = 64

test_wasserstein = true

X = rand(data_type, d, N)
Y_onehot = zeros(data_type, 1, N)
Y = zeros(data_type, d, N)
for i = 1:N
  Y[indmax(X[:,i]),i] = 1
  Y_onehot[i] = indmax(X[:,i])-1
end

# ground metric
M = zeros(d, d)
for i = 1:d
  for j = 1:d
    M[i,j] = abs(i-j)
  end
end


if use_cuda
  backend = GPUBackend()
else
  backend = CPUBackend()
end
init(backend)

data_layer = MemoryDataLayer(batch_size=50,
    data=Array[X, test_wasserstein ? Y : Y_onehot])

hidden_layers = map(1:n_hidden_layer) do i
  InnerProductLayer(name="ip$i", output_dim=n_hidden_unit, neuron=Neurons.ReLU(),
      tops=[symbol("ip$i")], bottoms=[i == 1 ? :data : symbol("ip$(i-1)")])
end
pred_layer = InnerProductLayer(name="pred", output_dim=d,
    tops=[:pred], bottoms=[symbol("ip$n_hidden_layer")])
if test_wasserstein
  softmax_layer = SoftmaxLayer(bottoms=[:pred], tops=[:softpred])
  loss_layer = WassersteinLossLayer(bottoms=[:softpred, :label],
      M = M, lambda = lambda, sinkhorn_iter = sinkhorn_iter)
else
  loss_layer = SoftmaxLossLayer(bottoms=[:pred, :label])
end

common_layers = [hidden_layers..., pred_layer]
if test_wasserstein
  common_layers = [common_layers..., softmax_layer]
end
net = Net("Training", backend, [data_layer, common_layers..., loss_layer])
params = SolverParameters(max_iter=10000,
    lr_policy=LRPolicy.Fixed(0.01), mom_policy=MomPolicy.Fixed(0.9))
solver = SGD(params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# for simplicity, we just use the training data as test...
data_layer_test = MemoryDataLayer(batch_size=100, data=Array[X, Y_onehot])
acc_layer = AccuracyLayer(bottoms=[:softpred, :label])
test_layers = [data_layer_test, common_layers..., acc_layer]
if !test_wasserstein
  softmax_layer = SoftmaxLayer(bottoms=[:pred], tops=[:softpred])
  test_layers = [test_layers..., softmax_layer]
end
test_net = Net("Testing", backend, test_layers)
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=500)

solve(solver, net)

shutdown(backend)
