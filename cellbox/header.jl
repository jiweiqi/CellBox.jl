### Parsing config
using ArgParse
using YAML

s = ArgParseSettings()
@add_arg_table s begin
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
else
    runtime = YAML.load_file("./runtime.yaml")
    expr_name = runtime["expr_name"]
end
is_restart = parsed_args["is-restart"] | Bool(runtime["is_restart"])
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

cd(dirname(@__DIR__))
ENV["GKSwstype"] = "100"
fig_path = string("./results/", expr_name, "/figs")
ckpt_path = string("./results/", expr_name, "/checkpoint")
config_path = "./results/$expr_name/config.yaml"

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
