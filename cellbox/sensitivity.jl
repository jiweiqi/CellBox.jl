# Following code can be plugged into cellbox.jl

begin
    function  loss_all(p)
        loss_epoch = zeros(Float32, n_exp);
        for i_exp in 1:n_exp
            loss_epoch[i_exp] = loss_neuralode(p, i_exp)
        end
        loss_total = mean(loss_epoch);
    end
    p_grad = Zygote.gradient(x -> loss_neuralode(x), p)[1]

    backend(:pyplot)

    xs = vcat(["α"], [string("W i=", i) for i = 1:ns])
    ys = [string("W j=", i) for i = 1:ns]
    plt = heatmap(xs, ys, abs.(p_grad) ./ maximum(abs.(p_grad)), aspect_ratio = 1);
    title!(plt, "Scaled sensitivity map p(α | W)");

    plt_p = heatmap(xs, ys, abs.(p), aspect_ratio = 1);
    title!(plt_p, "map p");

    plt_p_gold = heatmap(xs, ys, abs.(p_gold), aspect_ratio = 1);
    title!(plt_p_gold, "map p_gold");

    plt_p_diff = heatmap(xs, ys, abs.(p_gold - p), aspect_ratio = 1);
    title!(plt_p_diff, "map p_gold - p");

    plt_all = plot(plt, plt_p, plt_p_gold, plt_p_diff, layout=(2, 2), size=(1600, 1600));
    png(plt_all, string(fig_path, "/sens"))

    backend(:gr)
end
