cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0, p, i_exp)
    l_plt = []
    for i in 1:minimum([10, ns])
        plt = scatter(ts, ode_data[i,:], label="data");
        plot!(plt, ts, pred[i,:], label="pred");
        ylabel!(plt, "x$i")
        xlabel!(plt, "Time")
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false, size=(1000, 1000));
    png(plt_all, string(fig_path, "/conditions/i_exp_", i_exp))
    return false
end

cbp = function (p, iter)
    plt_alpha = scatter(p_gold[:,1],p[:,1])
    xlabel!("ground truth alpha")
    ylabel!("inferred alpha")
    plt_w = scatter([p_gold[:,2:end]...],[p[:,2:end]...])
    xlabel!("ground truth w")
    ylabel!("inferred w")
    plt_p = plot([plt_alpha, plt_w]..., legend=false, layout = grid(1, 2), size=(850, 400))
    png(plt_p, string(fig_path, "/p_inference_iter", iter))
end

l_loss_train = []
l_loss_val = []
l_loss_test = []
l_grad = []
l_loss_network = []

iter = 1
loss_val_movingavg = Inf
no_change = 0

cb = function (p, loss_train, loss_val, loss_test, g_norm, loss_p)
    global l_loss_train, l_loss_val, l_grad, l_loss_network, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_loss_test, loss_test)
    push!(l_loss_network, loss_p)
    push!(l_grad, g_norm)
    @save string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val l_loss_test l_grad l_loss_network iter

    if (iter % n_plot == 0)
        if parsed_args["disable-display"]
            println([iter loss_train loss_val loss_test g_norm loss_p])
        else
            l_exp = randperm(n_exp)[1:1]
            println("update plot for ", l_exp)
            for i_exp in l_exp
                cbi(p, i_exp)
            end

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
            cbp(p, "_tracking")
        end
    end
    iter += 1;
end

if is_restart
    @load string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val l_grad iter l_loss_test l_loss_network;
    iter += 1;
end

checkconvergence = function ()
    # TODO: write in log file rather than saved in RAM
    global loss_val_movingavg, no_change
    loss_val_movingavg_new = mean(l_loss_val[max(1, end-n_iter_buffer):end])  # TODO: no check bounds

    if (iter > n_iter_burnin) & (loss_val_movingavg < loss_val_movingavg_new)
        no_change += 1
        if (no_change > n_iter_tol) | (iter > n_iter_max)
            return true
        end
    else
        no_change = 0
        loss_val_movingavg = loss_val_movingavg_new
    end
    return false
end