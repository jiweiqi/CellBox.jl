epochs = ProgressBar(iter:n_iter_max);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
for epoch in epochs
    global p
    if conf["epoch_size"] != -1
        epoch_samples = sample(1:n_exp_train, conf["epoch_size"], replace=false)
    else
        epoch_samples = 1:n_exp_train
    end
    for i_exp in epoch_samples
    # for i_exp in randperm(n_exp_train)
        batch = rand(batch_size:ntotal)
        grad = Zygote.gradient(x -> loss_neuralode(x, i_exp, batch), p)[1]
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train+1 : n_exp_train+n_exp_val]);
    loss_test = mean(loss_epoch[n_exp_train+n_exp_val+1:end]);
    g_norm = mean(grad_norm)
    loss_p = loss_network(p)

    set_description(epochs, string(@sprintf("Loss train %.4e tol %d/%d lr %.1e",
                            loss_train, no_change, n_iter_tol, opt[1].eta)))
    cb(p, loss_train, loss_val, loss_test, g_norm, loss_p);

    if n_iter_tol > 0
        if checkconvergence()
            break
        end
    end
end

if !parsed_args["disable-display"]
    for i_exp in n_exp_train:n_exp
        cbi(p, i_exp)
    end

    if ispath(string(fig_path, "/p_inference_iter_tracking.png"))
        rm(string(fig_path, "/p_inference_iter_tracking.png"))
    end
    cbp(p, iter)
end
