
l_loss_train = []
l_loss_val = []
l_loss_test = []
l_grad = []
l_loss_network = []

iter = 1
loss_val_movingavg = Inf
no_change = 0

cb = function (p, loss_train, loss_val, loss_test, g_norm)
    global l_loss_train, l_loss_val, l_grad, l_loss_network, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_loss_test, loss_test)

    push!(l_loss_network, loss_network(p))
    push!(l_grad, g_norm)
    @save string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val l_loss_test l_grad l_loss_network iter

    @suppress_err if (iter % n_plot == 0)
        if parsed_args["disable-display"]
            println([iter loss_train loss_val loss_test g_norm loss_p])
        else
            l_exp = randperm(n_exp)[1:1]
            println("update plot for ", l_exp)
            for i_exp in l_exp
                cbi(p, i_exp)
            end
            cbloss()
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
