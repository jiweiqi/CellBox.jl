using YAML

if !(@isdefined expr_name) | !(ispath(expr_name))
    error("Variable expr_name not specified \n or the folder does not exist\n")
end
println("runing $expr_name ...\n")

conf = YAML.load_file("$expr_name/config.yaml")
is_restart = Bool(conf["is_restart"])

include("header.jl")

ns = Int64(conf["ns"]); # number of nodes / species
tfinal = Float64(conf["tfinal"]);
ntotal = Int64(conf["ntotal"]);  # number of samples for each perturbation
batch_size = Int64(conf["batch_size"]);  # STEER

n_exp_train = Int64(conf["n_exp_train"]);
n_exp_val = Int64(conf["n_exp_val"]);
n_exp_test = Int64(conf["n_exp_test"]);

n_exp = n_exp_train + n_exp_val + n_exp_test;
noise = Float64(conf["noise"])
opt = ADAMW(Float64(conf["lr"]), (0.9, 0.999), Float64(conf["weight_decay"]));

n_iter_max = Int64(conf["n_iter_max"])
n_plot = Int64(conf["n_plot"]);  # frequency of callback
n_iter_buffer = Int64(conf["n_iter_buffer"])
n_iter_burnin = Int64(conf["n_iter_burnin"])
n_iter_tol = Int64(conf["n_iter_tol"])
convergence_tol = Float64(conf["convergence_tol"])

Random.seed!(1);

# Generate data sets
# μ_list = rand(n_exp, ns);  #random sampling
μ_list = randomLHC(n_exp, ns) ./ n_exp;  # random LHS sampling
nμ = Int64(conf["n_mu"])
for i = 1:n_exp
    ind_zero = randperm(ns)[nμ+1:ns]
    μ_list[i, ind_zero] .= 0
end
# TODO: negative μ?

include("network.jl")
include("callback.jl")
include("train.jl")
