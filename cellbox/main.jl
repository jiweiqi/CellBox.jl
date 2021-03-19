include("header.jl")
include("visual.jl")


opt = ADAMW(Float64(conf["lr"]), (0.9, 0.999), Float64(conf["weight_decay"]));

include("network.jl")

if Bool(conf["is_nn"])
    include("network_nn.jl")
end

include("callback.jl")
include("train.jl")

@suppress_err begin
    include("export.jl")
end
