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
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

l_loss_train = []
l_loss_val = []
l_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global l_loss_train, l_loss_val, l_grad, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_grad, g_norm)

    if iter % n_plot == 0

        l_exp = randperm(n_exp)[1:1]
        println("update plot for ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, yscale=:log10, label="train")
        plot!(plt_loss, l_loss_val, yscale=:log10, label="val")
        plt_grad = plot(l_grad, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        # ylims!(plt_loss, (-Inf, 1))
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter;
    end
    iter += 1;
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter;
    iter += 1;
end

checkconvergence = function ()
    is_stop = false
    # TODO: write in log file rather than saved in RAM
    # loss_val_movingavg_new = loss_val[end-n_iter_buffer:end]  # TODO: no check bounds
    # if iter > n_iter_burnin & loss_val_movingavg > loss_val_movingavg_new
    #     no_change += 1
    #     if no_change > n_iter_tol | iter > n_iter_max
    #         # TODO: end training
    #         # maybe return a boolean for cb() and end in the main for loop
    #     end
    # else
    #     loss_val_movingavg = loss_val_movingavg_new
    # end
    return is_stop
end