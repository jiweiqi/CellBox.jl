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

if ispath("figs") == false
    mkdir("figs")
end
if ispath("checkpoint") == false
    mkdir("checkpoint")
end

fig_path = string("figs/", expr_name)
ckpt_path = string("checkpoint/", expr_name)

if ispath(fig_path) == false
    mkdir(fig_path)
else
    if !is_restart
        rm(fig_path, recursive=true)
    end
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
else
    if !is_restart
        rm(ckpt_path, recursive=true)
    end
end
