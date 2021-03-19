cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    _ts = 0:ts[end]/nplot:ts[end]
    pred = predict_neuralode(u0, p, i_exp, nothing, false)
    gold = predict_neuralode(u0, p_gold, i_exp, nothing, false)
    l_plt = []
    for i in 1:minimum([10, ns])
        plt = Plots.scatter(ts, ode_data[i,:], label="Data");
        plot!(plt, _ts, pred[i,:], label="Prediction");
        plot!(plt, _ts, gold[i,:], label="Ground truth");
        ylabel!(plt, "x$i")
        xlabel!(plt, "Time")
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false, size=(1000, 1000));
    png(plt_all, string(fig_path, "/conditions/i_exp_", i_exp))
    return false
end

cbi! = function (p, i_exp, plt)
    ode_data = ode_data_list[i_exp, :, :]
    _ts = 0:ts[end]/nplot:ts[end]
    pred = predict_neuralode(u0, p, i_exp, nothing, false)
    gold = predict_neuralode(u0, p_gold, i_exp, nothing, false)
    ylabel!(plt, "Protein levels")
    xlabel!(plt, "Time")
    plot!(plt, _ts, gold[1,:], line=:dash, label="Ground truth", color=palette(:tab10)[1]);
    Plots.scatter!(plt, ts, ode_data[1,:], label="Data", color=palette(:tab10)[1]);
    plot!(plt, _ts, pred[1,:], label="Prediction", color=palette(:tab10)[1]);
    for i in 2:minimum([5, ns])
        plot!(plt, _ts, gold[i,:], line=:dash, label=nothing, color=palette(:tab10)[i]);
        Plots.scatter!(plt, ts, ode_data[i,:], label=nothing, color=palette(:tab10)[i]);
        plot!(plt, _ts, pred[i,:], label=nothing, color=palette(:tab10)[i]);
    end
    png(plt, string(fig_path, "/trajectories_cond_", i_exp, ".png"))
end

cbp = function (p, iter)
    plt_alpha = Plots.scatter(p_gold[:,1],p[:,1])
    xlabel!("ground truth alpha")
    ylabel!("inferred alpha")
    plt_w = Plots.scatter([p_gold[:,2:end]...],[p[:,2:end]...])
    xlabel!("ground truth w")
    ylabel!("inferred w")
    plt_p = plot([plt_alpha, plt_w]..., legend=false, layout = grid(1, 2), size=(850, 400))
    png(plt_p, string(fig_path, "/p_inference_iter", iter))
end

cbloss = function ()
    plt_loss = plot(l_loss_train, yscale=:log10, label="train")
    plot!(plt_loss, l_loss_val, yscale=:log10, label="val")
    plot!(plt_loss, l_loss_test, yscale=:log10, label="test")
    plt_grad = plot(l_grad, yscale=:log10, label="grad_norm")
    plt_p = plot([l_loss_network[i][1] for i in 1:length(l_loss_network)], label="alpha")
    plot!(plt_p, [l_loss_network[i][2] for i in 1:length(l_loss_network)], label="w")
    xlabel!(plt_loss, "Epoch")
    xlabel!(plt_grad, "Epoch")
    xlabel!(plt_p, "Epoch")
    ylabel!(plt_loss, "Loss")
    ylabel!(plt_grad, "Gradient Norm")
    ylabel!(plt_p, "Parameter correlation")
    # ylims!(plt_loss, (-Inf, 1))
    plt_all = plot([plt_loss, plt_grad, plt_p]..., legend=:top, layout = grid(3, 1), size=(600, 900))
    png(plt_all, string(fig_path, "/loss_grad"))
end
