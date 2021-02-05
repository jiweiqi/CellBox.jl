is_restart = false;
expr_name = "Test_01";
include("header.jl")

# Arguments
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
n_plot = 20;  # frequency of callback
n_iter_buffer = 50
n_iter_burnin = 100
n_iter_tol = 10
convergence_tol = 1e-8

# Generate data sets
# μ_list = rand(n_exp, ns);  #random sampling
μ_list = randomLHC(n_exp, ns) ./ n_exp;  # random LHS sampling
# TODO: sparsity for μ_list too
# TODO: negative μ?

p_gold = gen_network(ns, weight_params=(-1.0, 1.0), sparsity=0.9, drop_range=(-0.1, 0.1))
p = gen_network(ns; weight_params=(0.0, 0.01), sparsity=0)
# show_network(p)

include("network.jl")
include("callback.jl")
include("train.jl")
