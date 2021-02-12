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
using ArgParse
cd(dirname(@__DIR__))
s = ArgParseSettings()

@add_arg_table s begin
    "--disable-display"
        help = "Use for UNIX server"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

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
