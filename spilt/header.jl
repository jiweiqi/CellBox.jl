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

fig_path = string(expr_name, "/figs")
ckpt_path = string(expr_name, "/checkpoint")

if ispath(fig_path) == false
    mkdir(fig_path)
else
    if !is_restart
        rm(fig_path, recursive=true)
        mkdir(fig_path)
    end
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
else
    if !is_restart
        rm(ckpt_path, recursive=true)
        mkdir(fig_path)
    end
end
