

α = abs.(rand(Uniform(-1.0, 1.0), (ns))) .+ 0.5

n_nodes = ns;
nn_W = Chain(x -> x,
              Dense(ns, n_nodes, gelu),
              Dense(n_nodes, n_nodes, gelu),
              Dense(n_nodes, n_nodes, gelu),
              Dense(n_nodes, n_nodes, gelu),
              Dense(n_nodes, ns))
pW, re = Flux.destructure(nn_W);

p = vcat(α, pW)

function cellbox!(du, u, p, t)
    Wu = re(@view(p[ns+1:end]))(u)
    pα = @view(p[1:ns])
    # du = Wu - μ - u
    @. du = pα * tanh(Wu - μ) - u
end

tspan = (0, tfinal);
ts = 0:tspan[2]/ntotal:tspan[2];
ts = ts[2:end];
u0 = zeros(ns);
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

predict_neuralode(u0, p, 1);
@show loss_neuralode(p, 1);

cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    _ts = 0:ts[end]/nplot:ts[end]
    pred = predict_neuralode(u0, p, i_exp, nothing, false)
    # gold = predict_neuralode(u0, p_gold, i_exp, nothing, false)
    l_plt = []
    for i in 1:minimum([10, ns])
        plt = Plots.scatter(ts, ode_data[i,:], label="Data");
        plot!(plt, _ts, pred[i,:], label="Prediction");
        # plot!(plt, _ts, gold[i,:], label="Ground truth");
        ylabel!(plt, "x$i")
        xlabel!(plt, "Time")
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false, size=(1000, 1000));
    png(plt_all, string(fig_path, "/conditions/i_exp_", i_exp))
    return false
end

function show_network(p)
    nothing
end

function loss_network(p)
     return 1.0, 1.0
end

cbp = function (p, iter)
    nothing
end

cbi(p, 1)
