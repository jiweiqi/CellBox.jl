# Random.seed!(1);
if "data" in keys(conf)
    idx_order = randperm(n_exp)
    pert = DataFrame(CSV.File(string(conf["data"],"/pert.csv"); header=false))
    μ_list = convert(Matrix,pert)[idx_order,:]
else
    μ_list = randomLHC(n_exp, ns) ./ n_exp;
    nμ = Int64(conf["n_mu"])
    for i = 1:n_exp
        nonzeros = findall(μ_list[i, :].>0)
        ind_zero = sample(nonzeros, max(0, length(nonzeros)-nμ), replace=false)
        μ_list[i, ind_zero] .= 0
    end
end

function gen_network(m; weight_params=(-1., 1.), sparsity=0., drop_range=(-1e-1, 1e-1))

    # uniform random for W matrix
    w = rand(Uniform(weight_params[1], weight_params[2]), (m, m))

    # Drop small values
    @inbounds for i in eachindex(w)
        w[i] = ifelse(drop_range[1]<=w[i]<=drop_range[2], 0, w[i])
    end

    # Add sparsity
    p = [sparsity, 1 - sparsity]
    w .*= sample([0, 1], weights(p), (m, m), replace=true)

    # Add α vector
    α = abs.(rand(Uniform(weight_params[1], weight_params[2]), (m))) .+ 0.5

    return hcat(α, w)
end

if "network" in keys(conf)
    df = DataFrame(CSV.File(conf["network"]))
    nodes = names(df)
    w = convert(Matrix, df[:,2:end])
    @assert size(w)[1] == size(w)[2]
    @assert size(w)[1] == ns
    if "randomize_network" in keys(conf)
        w_rand = rand(Normal(1, conf["randomize_network"]), (ns, ns))
        w = w .* w_rand
    end
    if "alpha" in keys(conf)
        α = ones(ns) .* conf["alpha"]
    else
        α = ones(ns) .* 0.2
    end
    p_gold = hcat(α, w)
elseif "data" in keys(conf)
    p_gold = hcat(ones(ns),zeros(ns,ns))
else
    p_gold = gen_network(ns, weight_params=(-1.0, 1.0),
                         sparsity=Float64(conf["sparsity"]),
                         drop_range=(Float64(conf["drop_range"]["lb"]), Float64(conf["drop_range"]["ub"])));
end
p = gen_network(ns; weight_params=(0.0, 0.01), sparsity=0);
# show_network(p)

function show_network(p)
    println("p_gold")
    show(stdout, "text/plain", round.(p_gold, digits=2))
    println("\np_learned")
    show(stdout, "text/plain", round.(p, digits=2))
end

function loss_network(p)
     # distalpha = cosine_dist(p_gold[:,1],p[:,1])
     # distw = cosine_dist(p_gold[:,2:end],p[:,2:end])
     @inbounds coralpha = cor(p_gold[:,1],p[:,1])
     @inbounds corw = cor([p_gold[:,2:end]...],[p[:,2:end]...])
     return coralpha, corw
end

function cellbox!(du, u, p, t)
    @inbounds du .= @view(p[:, 1]) .* tanh.(@view(p[:, 2:end]) * u - μ) .- u
end

tspan = (0, tfinal);
ts = 0:tspan[2]/ntotal:tspan[2];
ts = ts[2:end];
u0 = zeros(ns);
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2)
end

if "data" in keys(conf)
    expr = DataFrame(CSV.File(string(conf["data"],"/expr.csv"); header=false))
    ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
    ode_data_list[:,:,1] = convert(Matrix,expr)[idx_order,:]
else
    ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
    yscale_list = [];
    for i = 1:n_exp
        global μ = μ_list[i, 1:ns]
        ode_data = Array(solve(prob, Tsit5(), u0=u0, p=p_gold))

        ode_data += randn(size(ode_data)) .* noise
        ode_data_list[i, :, :] = ode_data

        push!(yscale_list, max_min(ode_data))
    end
    yscale = maximum(hcat(yscale_list...), dims=2);
end

function predict_neuralode(u0, p, i_exp=1, batch=ntotal, saveat=true)
    global μ = μ_list[i_exp, 1:ns]
    if saveat
        @inbounds _prob = remake(prob, p=p, tspan=[0, ts[batch]])
        pred = Array(solve(_prob, Tsit5(), saveat=ts[1:batch],
                     sensealg=InterpolatingAdjoint()))
    else # for full trajectory plotting
        @inbounds _prob = remake(prob, p=p, tspan=[0, ts[end]])
        pred = Array(solve(_prob, Tsit5(), saveat=0:ts[end]/nplot:ts[end]))
    end
    return pred
end
predict_neuralode(u0, p, 1);

function loss_neuralode(p, i_exp=1, batch=ntotal)
    pred = predict_neuralode(u0, p, i_exp, batch)
    @inbounds loss = mae(@views(ode_data_list[i_exp, :, 1:batch]), pred)
    return loss
end
@show loss_neuralode(p, 1);
