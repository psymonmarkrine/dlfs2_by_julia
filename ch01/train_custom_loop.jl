import Random: randperm
import Printf: @sprintf

using Plots

include("../common/optimizer.jl") # SGD
include("../dataset/spiral.jl")
include("two_layer_net.jl") # TwoLayerNet


# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data()
model = TwoLayerNet(2, hidden_size, 3)
optimizer = SGD(learning_rate)

# 学習で使用する変数
data_size = size(x, 1)
max_iters = div(data_size, batch_size)
total_loss = 0
loss_count = 0
loss_list = zeros(0)

for epoch = 1:max_epoch
    # データのシャッフル
    idx = randperm(data_size)
    global x = x[idx,:]
    global t = t[idx,:]

    for iters = 1:max_iters
        idx_end = iters * batch_size
        idx_begin = idx_end - batch_size + 1
        batch_x = x[idx_begin:idx_end, :]
        batch_t = t[idx_begin:idx_end, :]

        # 勾配を求め、パラメータを更新
        loss = forward(model, batch_x, batch_t)
        backward(model)
        update(optimizer, model.params, model.grads)

        global total_loss += loss
        global loss_count += 1

        # 定期的に学習経過を出力
        if iters % 10 == 0
            avg_loss = total_loss / loss_count
            println("| epoch $epoch |  iter $iters / $max_iters | loss $(@sprintf("%.2f", avg_loss))")
            append!(loss_list, avg_loss)
            total_loss, loss_count = 0, 0
        end
    end
end

# 学習結果のプロット
plot(1:length(loss_list), loss_list, label="train", xlabel="iterations (x10)", ylabel="loss")
savefig("../image/ch01/fig01-32.png")

# 境界領域のプロット
h = 0.001
x_range = (minimum(x[:, 1]) - 0.1):h:(maximum(x[:, 1]) + 0.1)
y_range = (minimum(x[:, 2]) - 0.1):h:(maximum(x[:, 2]) + 0.1)
xx = repeat(x_range, inner=length(y_range))
yy = repeat(y_range, outer=length(x_range))
X = hcat(xx, yy)
score = predict(model, X)
predict_cls = [i[2] for i=argmax(score, dims=2)]
Z = reshape(predict_cls, length(y_range), length(x_range))
contour(x_range, y_range, Z, fill=true, cbar=false, xticks=nothing, yticks=nothing, leg=false)


# データ点のプロット
N = 100
CLS_NUM = 3
markers = [:circle, :x, :utriangle]
for i = 1:CLS_NUM
    indx = t[:,i].==1
    scatter!(x[indx, 1], x[indx, 2], marker=markers[i])
end
savefig("../image/ch01/fig01-33.png")
