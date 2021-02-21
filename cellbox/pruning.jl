
p_cutoff = 0.5
pp = p
W = pp[:, 2:end];
W[findall(abs.(W) .< p_cutoff)] .= 0.0
pp[:, 2:end] .= W;

for i_exp in 1:n_exp
    cbi(pp, i_exp)
end
cbp(pp, iter)
