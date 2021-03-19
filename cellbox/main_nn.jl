include("header.jl")

# Random.seed!(1);
μ_list = randomLHC(n_exp, ns) ./ n_exp;
nμ = Int64(conf["n_mu"])
for i = 1:n_exp
    nonzeros = findall(μ_list[i, :].>0)
    ind_zero = sample(nonzeros, max(0, length(nonzeros)-nμ), replace=false)
    μ_list[i, ind_zero] .= 0
end

include("network.jl")
include("callback.jl")

opt = ADAMW(Float64(conf["lr"]), (0.9, 0.999), Float64(conf["weight_decay"]));
include("train.jl")

@suppress_err begin
    include("export.jl")
end
