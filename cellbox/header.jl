### Parsing config
using ArgParse
using YAML

s = ArgParseSettings()
@add_arg_table! s begin
    "--disable-display"
        help = "Use for UNIX server"
        action = :store_true
    "--expr_name"
        help = "Define expr name"
        required = false
    "--is-restart"
        help = "Continue training?"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

if parsed_args["expr_name"] != nothing
    expr_name = parsed_args["expr_name"]
    is_restart = parsed_args["is-restart"]
else
    runtime = YAML.load_file("./runtime.yaml")
    expr_name = runtime["expr_name"]
    is_restart = Bool(runtime["is_restart"])
end
conf = YAML.load_file("$expr_name/config.yaml")


## Prepare for working env
using OrdinaryDiffEq, Flux, Optim, Random, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae
using Distributions
using StatsBase
using LatinHypercubeSampling
using BSON: @save, @load
using CSV
using DataFrames
using DelimitedFiles
using Plots.PlotMeasures
using Suppressor

cd(dirname(@__DIR__))
ENV["GKSwstype"] = "100"
fig_path = string("./results/", expr_name, "/figs")
ckpt_path = string("./results/", expr_name, "/checkpoint")
config_path = "./results/$expr_name/config.yaml"
pyplot()

if is_restart
    println("Continue to run $expr_name ...\n")
else
    println("Runing $expr_name ...\n")
end

fig_path = string(expr_name, "/figs")
ckpt_path = string(expr_name, "/checkpoint")

if !is_restart
    if ispath(fig_path)
        rm(fig_path, recursive=true)
    end
    if ispath(ckpt_path)
        rm(ckpt_path, recursive=true)
    end
end

if ispath(fig_path) == false
    mkdir(fig_path)
    mkdir(string(fig_path, "/conditions"))
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end

if haskey(conf, "grad_max")
    grad_max = conf["grad_max"]
else
    grad_max = Inf
end

ns = Int64(conf["ns"]); # number of nodes / species
tfinal = Float64(conf["tfinal"]);
ntotal = Int64(conf["ntotal"]);  # number of samples for each perturbation
nplot = Int64(conf["nplot"]);
batch_size = Int64(conf["batch_size"]);  # STEER

n_exp_train = Int64(conf["n_exp_train"]);
n_exp_val = Int64(conf["n_exp_val"]);
n_exp_test = Int64(conf["n_exp_test"]);

n_exp = n_exp_train + n_exp_val + n_exp_test;
noise = Float64(conf["noise"])
opt = ADAMW(Float64(conf["lr"]), (0.9, 0.999), Float64(conf["weight_decay"]));

n_iter_max = Int64(conf["n_iter_max"])
n_plot = Int64(conf["n_plot"]);
n_iter_buffer = Int64(conf["n_iter_buffer"])
n_iter_burnin = Int64(conf["n_iter_burnin"])
n_iter_tol = Int64(conf["n_iter_tol"])
convergence_tol = Float64(conf["convergence_tol"])
