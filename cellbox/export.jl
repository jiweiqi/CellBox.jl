# Plot and save the trained params
params = hcat(nodes[2:end], p[:,2:end])
CSV.write(string(ckpt_path,"/params.csv"), DataFrame(params), header=nodes)
if !parsed_args["disable-display"]
    if ispath(string(fig_path, "/p_inference_iter_tracking.png"))
        rm(string(fig_path, "/p_inference_iter_tracking.png"))
    end
    cbp(p, iter)
end

if "data" in keys(conf)
    cbrd(p)
end


# Plot for each conditions
if !parsed_args["disable-display"]
   for i in n_exp-2:n_exp
      plt = plot(size=(500, 300), legend=:right)
      cbi!(p, i, plt)
   end
end


# Plot comparison heatmaps
if !parsed_args["disable-display"]
    l = @layout [a{0.457w} b{0.457w}]
    pyplot_ticks = [string("#", g, "#") for g in nodes[2:end]]
    heatmap1 = heatmap(p_gold[:, 2:end], legend=false, size=(320, 320),
                       c=:ice, xtickfontrotation=90)
    title!("Ground truth")
    xticks!(1:length(nodes)-1, pyplot_ticks)
    yticks!(1:length(nodes)-1, pyplot_ticks)
    heatmap2 = heatmap(p[:, 2:end], size=(360, 320), c=:ice,
                       clim=(-1, 1), xtickfontrotation=90)
    title!("Training result")
    xticks!(1:length(nodes)-1, pyplot_ticks)
    yticks!(1:length(nodes)-1, pyplot_ticks)
    heatmaps = plot(heatmap1, heatmap2, layout = l, size=(700, 320))
    png(heatmaps, string(fig_path, "/heatmap.png"))
end

# Joint figure
if !parsed_args["disable-display"]
    pick_idx = n_exp
    plt = plot(size=(360, 320), legend=:right, leftmargin=40px, rightmargin=40px)
    cbi!(p, pick_idx, plt)
    title!("ODE trajectory")
    l = @layout [a{0.3w} b{0.3w} c{0.4w}]
    merged = plot(heatmap1, heatmap2, plt, layout = l, size=(1080, 320))
    png(merged, string(fig_path, "/merged.png"))
end
