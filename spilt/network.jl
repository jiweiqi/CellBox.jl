function show_network(p)
    println("p_gold")
    show(stdout, "text/plain", round.(p_gold, digits=2))
    println("\np_learned")
    show(stdout, "text/plain", round.(p, digits=2))
end

function cellbox!(du, u, p, t)
    du .= tanh.(view(p, :, 2:ns + 1) * u - μ) - view(p, :, 1) .* u
end

tspan = (0, tfinal);
ts = 0:tspan[2] / ntotal:tspan[2];
ts = ts[2:end];
u0 = zeros(ns);
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2)
end

ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
yscale_list = [];
for i = 1:n_exp
    global μ = μ_list[i, 1:ns]
    ode_data = Array(solve(prob, Tsit5(), u0=u0, p=p_gold))

    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data

    push!(yscale_list, max_min(ode_data))
end
yscale = maximum(hcat(yscale_list...), dims=2);

function predict_neuralode(u0, p, i_exp=1, batch=ntotal)
    global μ = μ_list[i_exp, 1:ns]
    _prob = remake(prob, p=p, tspan=[0, ts[batch]])
    pred = Array(solve(_prob, Tsit5(), saveat=ts[1:batch], sensealg=InterpolatingAdjoint()))
    return pred
end
predict_neuralode(u0, p, 1);

function loss_neuralode(p, i_exp=1, batch=ntotal)
    pred = predict_neuralode(u0, p, i_exp, batch)
    loss = mae(@views(ode_data_list[i_exp, :, 1:batch]), pred)
    return loss
end
@show loss_neuralode(p, 1);