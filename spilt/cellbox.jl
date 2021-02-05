is_restart = false;
expr_name = "Test_01";
include("header.jl")

# Arguments
n_epoch = 10000;
n_plot = 20;  # frequency of callback

ns = 5;  # number of nodes / species
tfinal = 20.0;
ntotal = 20;  # number of samples for each perturbation
batch_size = 8;  # STEER

n_exp_train = 20;
n_exp_val = 10;
n_exp = n_exp_train + n_exp_val;
noise = 0.01;
opt = ADAMW(5.f-3, (0.9, 0.999), 1.f-4);

n_iter_max = 1000
n_iter_buffer = 50
n_iter_burnin = 100
n_iter_tol = 10
convergence_tol = 1e-5
loss_val_movingavg = Inf

# Generate data sets
# μ_list = rand(n_exp, ns);  #random sampling
μ_list = randomLHC(n_exp, ns) ./ n_exp;  # random LHS sampling
# TODO: sparsity for μ_list too
# TODO: hypertube  sampling

function gen_network(m; weight_params=(0., 1.), sparsity=0.)
    # TODO: network constraints: range and zeros
    w = rand(Uniform(weight_params[1], weight_params[2]), (m, m))
    p = [sparsity, 1 - sparsity]
    w .*= sample([0, 1], weights(p), (m, m), replace=true)
    α = abs.(rand(Uniform(weight_params[1], weight_params[2]), (m)))
    return hcat(α, w)
end

p_gold = gen_network(ns; weight_params=(0.0, 1.0), sparsity=0.9);
p = gen_network(ns; weight_params=(0.0, 0.01), sparsity=0);

include("network.jl")
include("callback.jl")
# opt = ADAMW(1.f-4, (0.9, 0.999), 1.f-6);
include("train.jl")

# show_network(p) # TODO: a network inference loss
