epochs = ProgressBar(iter:n_iter_max);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        batch = rand(batch_size:ntotal)
        grad = Zygote.gradient(x -> loss_neuralode(x, i_exp, batch), p)[1]
        grad_norm[i_exp] = norm(grad, 2)
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

    set_description(epochs, string(@sprintf("Loss train %.4e Tolerance %d/%d", loss_train, no_change, n_iter_tol)))
    cb(p, loss_train, loss_val, loss_test, g_norm, loss_p);

    if checkconvergence()
        break
    end
end

for i_exp in 1:n_exp
    cbi(p, i_exp)
end

rm(string(fig_path, "/p_inference_iter_tracking.png"))
cbp(p, iter)
