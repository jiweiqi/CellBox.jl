# Plot for each conditions
if !parsed_args["disable-display"]
    for i_exp in n_exp_train:n_exp
        cbi(p, i_exp)
    end
end

for i in n_exp-2:n_exp
   cbi!(p, i)
end

# Plot and save the trained params
params = hcat(nodes[2:end], p[:,2:end])
CSV.write(string(ckpt_path,"/params.csv"), DataFrame(params), header=nodes)
if !parsed_args["disable-display"]
    if ispath(string(fig_path, "/p_inference_iter_tracking.png"))
        rm(string(fig_path, "/p_inference_iter_tracking.png"))
    end
    cbp(p, iter)
end

# Plot comparison heatmaps
if !parsed_args["disable-display"]
    l = @layout [
       a{0.44w} b
    ]
    heatmap1 = heatmap(p[:, 2:end], legend=false, size=(320, 320), c=:ice)
    title!("Ground truth")
    xticks!(1:length(nodes)-1, nodes[2:end])
    yticks!(1:length(nodes)-1, nodes[2:end])
    heatmap2 = heatmap(p_gold[:, 2:end], size=(360, 320), c=:ice)
    title!("Training results")
    xticks!(1:length(nodes)-1, nodes[2:end])
    yticks!(1:length(nodes)-1, nodes[2:end])
    heatmaps = plot(heatmap1, heatmap2, layout = l, size=(680, 300))
    png(heatmaps, string(fig_path, "/heatmap.png"))
end
