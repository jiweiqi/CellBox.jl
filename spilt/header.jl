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

global expr_name
cd(dirname(@__DIR__))

if ispath("figs") == false
    mkdir("figs")
end
if ispath("checkpoint") == false
    mkdir("checkpoint")
end

fig_path = string("figs/", expr_name)
ckpt_path = string("checkpoint/", expr_name)

if !is_restart
    rm(fig_path, recursive=true)
    rm(ckpt_path, recursive=true)
end

if ispath(fig_path) == false
    mkdir(fig_path)
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end
