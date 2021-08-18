using Plots
include("../dataset/spiral.jl")


x, t = load_data()
println("x $(size(x))")  # (300, 2)
println("t $(size(t))")  # (300, 3)

# データ点のプロット
N = 100
CLS_NUM = 3
markers = [:circle, :x, :utriangle]
plot(xlim=(-1,1), ylim=(-1,1), leg=false)
for i = 1:CLS_NUM
    indx = t[:,i].==1
    scatter!(x[indx, 1], x[indx, 2], marker=markers[i])
end
savefig("../image/ch01/fig01-31.png")
